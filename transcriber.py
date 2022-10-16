import enum
import logging
import os
import queue
from threading import Thread
from typing import Callable, Optional
import numpy as np
import pathlib
import ctypes
from scipy.io import wavfile
import sys

from wurlitzer import pipes
import pyaudio

import auditok

sr = 16000
sw = 2
ch = 1
eth = 55 # alias for energy_threshold, default value is 50

# When the app is opened as a .app from Finder, the path doesn't contain /usr/local/bin
# which breaks the call to run `ffmpeg`. This sets the path manually to fix that.
os.environ["PATH"] += os.pathsep + "/usr/local/bin"

# this needs to match the C struct in whisper.h
class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy",             ctypes.c_int),
        ("n_threads",            ctypes.c_int),
        ("offset_ms",            ctypes.c_int),
        ("translate",            ctypes.c_bool),
        ("no_context",           ctypes.c_bool),
        ("print_special_tokens", ctypes.c_bool),
        ("print_progress",       ctypes.c_bool),
        ("print_realtime",       ctypes.c_bool),
        ("print_timestamps",     ctypes.c_bool),
        ("language",             ctypes.c_char_p),
        ("greedy",               ctypes.c_int * 1),
    ]


class Transcriber:
    """Transcriber records audio from a system microphone and transcribes it into text using Whisper."""

    class Task(enum.Enum):
        TRANSLATE = "translate"
        TRANSCRIBE = "transcribe"

    current_thread: Optional[Thread]
    #current_stream: Optional[sounddevice.InputStream]
    is_running = False
    MAX_QUEUE_SIZE = 10

    def __init__(self, model_name: str, language: Optional[str],
                 text_callback: Callable[[str], None], task: Task) -> None:
        self.model_name = model_name

        self.libname = pathlib.Path().absolute() / "libwhisper.so"
        self.whisper = ctypes.CDLL(self.libname)
        # tell Python what are the return types of the functions
        self.whisper.whisper_init.restype                  = ctypes.c_void_p
        self.whisper.whisper_full_default_params.restype   = WhisperFullParams
        self.whisper.whisper_full_get_segment_text.restype = ctypes.c_char_p
         # initialize whisper.cpp context
        self.ctx = self.whisper.whisper_init(("models/ggml-" + self.model_name + ".bin").encode("utf-8"))
        # get default whisper parameters and adjust as needed
        self.params = self.whisper.whisper_full_default_params(0)
        self.params.print_realtime = True
        self.params.print_progress = False
        self.params.language = language.encode()
        if str(task) == "Task.TRANSLATE":
            self.params.translate = True
        else:
            self.params.translate = False
        self.params.n_threads = os.cpu_count() - 1


        #self.current_stream = None
        self.text_callback = text_callback
        self.language = language
        self.task = task
        self.queue: queue.Queue[np.ndarray] = queue.Queue(
            Transcriber.MAX_QUEUE_SIZE,
        )
        print("Running on ", self.params.n_threads, " threads.")

    def start_recording(self, block_duration=10, input_device_index: Optional[int] = None):

        self.block_duration=block_duration

        print("input_device_index is: ", input_device_index)

        self.is_running = True

        self.current_thread = Thread(target=self.process_queue)
        self.current_thread.start()

        self.recorder = Thread(target=self.record)
        self.recorder.start()

    def record(self):
        n = 0
        data = None

        for region in auditok.split(input=None, sr=sr, sw=sw, ch=ch, eth=eth):
            if data == None:
                data = region
            else:
                data = data + region
            if data.duration > self.block_duration:
                data.save(  f"tmp{n}.wav") # progress bar requires `tqdm`
                self.queue.put(n, block=False)
                data = None
                n = n + 1

    def process_queue(self):
        while self.is_running:
            try:
                block = self.queue.get(block=False)
                print(self.language)
                #print(block)
                outfile = open("transcript.txt", "a")
                logging.debug(
                    'Processing next frame. Current queue size: %d' % self.queue.qsize())

                wavpath=f"tmp{block}.wav"
                wavpath = str.encode(wavpath)

                # load WAV file
                samplerate, data = wavfile.read(wavpath)

                # convert to 32-bit float
                data = data.astype('float32')/32768.0

                # run the inference
                result = self.whisper.whisper_full(ctypes.c_void_p(self.ctx), self.params, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(data))
                if result != 0:
                    print("Error: {}".format(result))
                    exit(1)


                # print results from Python
                txtfull = ""
                print("\nResults from Python:\n")
                n_segments = self.whisper.whisper_full_n_segments(ctypes.c_void_p(self.ctx))
                for i in range(n_segments):
                    t0  = self.whisper.whisper_full_get_segment_t0(ctypes.c_void_p(self.ctx), i)
                    t1  = self.whisper.whisper_full_get_segment_t1(ctypes.c_void_p(self.ctx), i)
                    txt = self.whisper.whisper_full_get_segment_text(ctypes.c_void_p(self.ctx), i)
                    txtfull = txtfull + "\n" + txt.decode('utf-8')


                    print(f"{t0/1000.0:.3f} - {t1/1000.0:.3f} : {txt.decode('utf-8')}")
                os.remove(f"tmp{block}.wav")
                outfile.write(txtfull + "\n")


                logging.debug("Received next result: \"%s\"" % txtfull)
                try:
                  self.text_callback(txtfull)  # type: ignore
                except:
                    continue
            except queue.Empty:
                continue

    def get_device_sample_rate(self, device_id: Optional[int]) -> int:
        """Returns the sample rate to be used for recording. It uses the default sample rate
        provided by Whisper if the microphone supports it, or else it uses the device's default
        sample rate.
        """
        whisper_sample_rate = 16000
        try:
            sounddevice.check_input_settings(
                device=device_id, samplerate=whisper_sample_rate)
            return whisper_sample_rate
        except:
            device_info = sounddevice.query_devices(device=device_id)
            if isinstance(device_info, dict):
                return int(device_info.get('default_samplerate', whisper_sample_rate))
            return whisper_sample_rate

    def stream_callback(self, in_data, frame_count, time_info, status):
        # Try to enqueue the next block. If the queue is already full, drop the block.
        try:
            n = self.queue.qsize() + 1
            #sounddevice.write(f"temp{n}.wav", in_data)
            write(f"tmp{n}.wav", 16000, in_data)
            self.queue.put(n, block=False)
        except queue.Full:
            return

    def stop_recording(self):
        if self.recorder != None:
            #self.recorder.close()
            logging.debug('Closed recording stream')

        self.is_running = False
        self.queue.queue.clear()

        if self.current_thread != None:
            logging.debug('Waiting for processing thread to terminate')
            #self.current_thread.join()
            logging.debug('Processing thread terminated')
