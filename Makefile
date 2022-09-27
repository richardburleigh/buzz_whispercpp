buzz:
	pyinstaller --noconfirm Buzz.spec

clean:
	rm -r dist/* || true

bundle_mac:
	make buzz
	tar -czf dist/Buzz.zip dist/Buzz
	mkdir -p dist/dmg && cp -r dist/Buzz.app dist/dmg
	create-dmg \
		--volname "Buzz" \
		--volicon "dist/Buzz.app/Contents/Resources/icon-windowed.icns" \
		--window-pos 200 120 \
		--window-size 600 300 \
		--icon-size 100 \
		--icon "dist/Buzz.app/Contents/Resources/icon-windowed.icns" 175 120 \
		--hide-extension "Buzz.app" \
		--app-drop-link 425 120 \
		"dist/Buzz.dmg" \
		"dist/dmg/"

release:
	make clean
	make bundle_mac
	poetry version ${version}
	git tag "v${version}"
