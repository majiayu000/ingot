#!/usr/bin/env bash
# Build dist/Ingot.app and dist/Ingot-VERSION.dmg on Apple Silicon macOS.
# Needs Rust and the Xcode Command Line Tools.
#
# Without settings the app is signed ad hoc, which macOS blocks on other Macs
# until the user allows it in System Settings. For public downloads:
#   INGOT_SIGN_ID="Developer ID Application: NAME (TEAM)"   sign the app and image
#   INGOT_NOTARY_PROFILE=PROFILE   notarize with `notarytool store-credentials` PROFILE
# A signed build also writes dist/appcast.xml for Sparkle, signed with the
# EdDSA key that `generate_keys --account ingot` keeps in the login keychain.
# Attach both files to the GitHub release.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
[ "$(uname -s)-$(uname -m)" = Darwin-arm64 ] || { echo "package-app.sh must run on Apple Silicon macOS" >&2; exit 1; }

pkgid="$(cargo pkgid)"
version="${pkgid##*[#@]}"
target_dir="${CARGO_TARGET_DIR:-$root/target}"
dist="$root/dist"
app="$dist/Ingot.app"
dmg="$dist/Ingot-$version.dmg"
appcast="$dist/appcast.xml"
sparkle="$root/app/.build/artifacts/sparkle/Sparkle"
sign_id="${INGOT_SIGN_ID:--}"

cargo build --release --locked --bin ingot
swift build -c release --package-path app

work="$(mktemp -d)"
mount=""
cleanup() {
  [ -z "$mount" ] || hdiutil detach "$mount" -quiet || true
  rm -rf "${work:?}"
}
trap cleanup EXIT

swift app/make-art.swift icon "$work/icon.png"
swift app/make-art.swift background "$work/background.png"
iconset="$work/Ingot.iconset"
mkdir "$iconset"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$work/icon.png" --out "$iconset/icon_${size}x${size}.png" > /dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$work/icon.png" --out "$iconset/icon_${size}x${size}@2x.png" > /dev/null
done

mkdir -p "$dist"
rm -rf "${app:?}" "${dmg:?}" "${appcast:?}"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$app/Contents/Frameworks"
sed "s/VERSION/$version/g" app/Info.plist > "$app/Contents/Info.plist"
cp "app/.build/release/Ingot" "$app/Contents/MacOS/Ingot"
cp "$target_dir/release/ingot" "$app/Contents/Resources/ingot"
iconutil -c icns "$iconset" -o "$app/Contents/Resources/Ingot.icns"
cp LICENSE kernels/NOTICE "$app/Contents/Resources/"
cp "$sparkle/LICENSE" "$app/Contents/Resources/Sparkle-LICENSE"
ditto "$sparkle/Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework" "$app/Contents/Frameworks/Sparkle.framework"

options=(--force --sign "$sign_id")
[ "$sign_id" = - ] || options+=(--options runtime --timestamp)
codesign "${options[@]}" "$app/Contents/Resources/ingot"
# Sparkle's helpers first, inside out, as its documentation asks.
framework="$app/Contents/Frameworks/Sparkle.framework"
for part in XPCServices/Installer.xpc XPCServices/Downloader.xpc Autoupdate Updater.app; do
  codesign "${options[@]}" --preserve-metadata=entitlements "$framework/Versions/B/$part"
done
codesign "${options[@]}" "$framework"
codesign "${options[@]}" "$app"
codesign --verify --strict "$app"
# The app carries its own ticket too, so it opens offline once copied out.
if [ -n "${INGOT_NOTARY_PROFILE:-}" ]; then
  ditto -c -k --keepParent "$app" "$work/Ingot.zip"
  xcrun notarytool submit "$work/Ingot.zip" --keychain-profile "$INGOT_NOTARY_PROFILE" --wait
  xcrun stapler staple "$app"
fi

# A writable image first: the window layout lives in its .DS_Store.
image="$work/Ingot.dmg"
hdiutil create -quiet -size "$(($(du -sm "$app" | cut -f1) + 30))m" -fs HFS+ -volname Ingot "$image"
mount="$(hdiutil attach -nobrowse -noverify -noautoopen "$image" | sed -n 's|^.*\(/Volumes/.*\)$|\1|p' | tail -1)"
[ -d "$mount" ] || { echo "cannot mount the disk image" >&2; exit 1; }
cp -R "$app" "$mount/"
ln -s /Applications "$mount/Applications"
mkdir "$mount/.background"
cp "$work/background.png" "$mount/.background/background.png"
volume="$(basename "$mount")"
osascript <<SCRIPT
tell application "Finder"
  tell disk "$volume"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set the bounds of container window to {200, 120, 860, 520}
    set options to the icon view options of container window
    set arrangement of options to not arranged
    set icon size of options to 112
    set background picture of options to file ".background:background.png"
    set position of item "Ingot.app" of container window to {170, 195}
    set position of item "Applications" of container window to {490, 195}
    close
    open
    delay 1
    close
  end tell
end tell
SCRIPT
sync
hdiutil detach "$mount" -quiet
mount=""
hdiutil convert -quiet "$image" -format UDZO -imagekey zlib-level=9 -o "$dmg"

if [ "$sign_id" != - ]; then
  codesign --force --sign "$sign_id" --timestamp "$dmg"
fi
if [ -n "${INGOT_NOTARY_PROFILE:-}" ]; then
  xcrun notarytool submit "$dmg" --keychain-profile "$INGOT_NOTARY_PROFILE" --wait
  xcrun stapler staple "$dmg"
fi
if [ "$sign_id" != - ]; then
  "$sparkle/bin/generate_appcast" --account ingot \
    --download-url-prefix "https://github.com/majiayu000/ingot/releases/download/v$version/" \
    --link "https://github.com/majiayu000/ingot/releases/tag/v$version" "$dist"
fi
shasum -a 256 "$dmg"
