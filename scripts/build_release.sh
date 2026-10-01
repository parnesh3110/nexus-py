#!/usr/bin/env bash
# Builds a standalone Nexus.app and Nexus-<version>.dmg in dist/: Python, packages, code and models
# all inside the app, so it runs on any Apple Silicon Mac with macOS 13+, from any folder.
#
#   bash scripts/build_release.sh                   # ad-hoc signed: runs on this Mac
#   NEXUS_SIGN_ID="Developer ID Application: Your Name (TEAMID)" bash scripts/build_release.sh
#                                                   # signed for distribution (hardened runtime)
#   NEXUS_SIGN_ID=... NEXUS_NOTARY_PROFILE=nexus bash scripts/build_release.sh
#                                                   # + notarized by Apple and stapled: opens
#                                                   #   on other Macs without a Gatekeeper warning
# One-time setup for notarizing: xcrun notarytool store-credentials nexus --apple-id you@example.com
#                                  --team-id TEAMID   (needs a paid Apple Developer account)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "Run bash scripts/setup.sh first."; exit 1; }

VERSION="$($PY -c 'import nexus; print(nexus.__version__)')"
PYVER="$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
BASE="$($PY -c 'import sys; print(sys.base_prefix)')"
SITE=".venv/lib/python$PYVER/site-packages"
DIST="$ROOT/dist"
APP="$DIST/Nexus.app"
RES="$APP/Contents/Resources"
EXE="$APP/Contents/MacOS/Nexus"
echo "==> Nexus $VERSION, Python $PYVER"
rm -rf "$DIST"
mkdir -p "$APP/Contents/MacOS" "$RES"

echo "==> Python runtime"
rsync -a --exclude '__pycache__' --exclude 'include' --exclude 'share' \
  --exclude "lib/python$PYVER/test" --exclude "lib/python$PYVER/idlelib" --exclude "lib/python$PYVER/tkinter" \
  --exclude "lib/python$PYVER/turtledemo" --exclude "lib/python$PYVER/ensurepip" --exclude 'lib/tcl*' --exclude 'lib/tk*' \
  --exclude 'lib/libtcl*' --exclude 'lib/libtk*' --exclude "lib/python$PYVER/site-packages" \
  "$BASE/" "$RES/python/"

echo "==> Packages"
# Leading "/" = only top-level packages (onnxruntime has its own "transformers" subfolder, which stays)
EXCLUDES=(--exclude '__pycache__' --exclude '/pip' --exclude '/pip-*' --exclude '/pytest' --exclude '/_pytest'
          --exclude '/pytest-*' --exclude '/pluggy*' --exclude '/iniconfig*')
if $PY -c 'import fastembed' 2>/dev/null; then
  # Intent matching runs on fastembed (ONNX), so PyTorch stays out: ~500 MB smaller.
  # (The optional trained gesture model needs PyTorch, so a release build skips it.)
  EXCLUDES+=(--exclude '/torch' --exclude '/torch-*' --exclude '/torchgen' --exclude '/functorch' --exclude '/triton*'
             --exclude '/sentence_transformers*' --exclude '/transformers' --exclude '/transformers-*'
             --exclude '/sympy' --exclude '/sympy-*' --exclude '/networkx' --exclude '/networkx-*' --exclude '/mpmath*')
  echo "    fastembed found: leaving PyTorch out"
else
  echo "    fastembed not installed (uv pip install fastembed): PyTorch stays in, the app is ~500 MB bigger"
fi
rsync -a "${EXCLUDES[@]}" "$SITE/" "$RES/site-packages/"

echo "==> App code"
mkdir -p "$RES/app"
rsync -a --exclude '__pycache__' nexus "$RES/app/"

echo "==> Precompiling (the app never writes .pyc files into itself; this keeps startup fast)"
$PY -m compileall -q -j 0 "$RES/app" "$RES/site-packages" "$RES/python/lib/python$PYVER" >/dev/null || true

echo "==> Models (shipped inside, so nothing downloads on first launch except the optional local AI)"
mkdir -p "$RES/models"
cp models/hand_landmarker.task "$RES/models/"
$PY - "$RES/models" <<'EOF'
import sys
out = sys.argv[1]
from faster_whisper.utils import download_model
download_model("base.en", output_dir=f"{out}/whisper-base.en")
try:
    from fastembed import TextEmbedding
    TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", cache_dir=f"{out}/fastembed")
except ImportError:
    print("    (fastembed missing: the intent model downloads on first use instead)")
EOF

echo "==> Launcher"
INC="$BASE/include/python$PYVER"
LIB="$RES/python/lib/libpython$PYVER.dylib"
[ -f "$INC/Python.h" ] && [ -f "$LIB" ] || { echo "Python headers or library missing under $BASE"; exit 1; }
install_name_tool -id "@rpath/libpython$PYVER.dylib" "$LIB"
clang -O2 -Wall -DNEXUS_BUNDLED -o "$EXE" scripts/nexus_stub.c -I"$INC" "$LIB" \
  -Wl,-rpath,@executable_path/../Resources/python/lib
OLD="$(otool -L "$EXE" | awk '/libpython/ {print $1; exit}')"
if [ -n "$OLD" ] && [[ "$OLD" != @rpath/* ]]; then
  install_name_tool -change "$OLD" "@rpath/libpython$PYVER.dylib" "$EXE"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Nexus</string>
  <key>CFBundleDisplayName</key><string>Nexus</string>
  <key>CFBundleIdentifier</key><string>dev.parnesh.nexus</string>
  <key>CFBundleExecutable</key><string>Nexus</string>
  <key>CFBundleIconFile</key><string>Nexus</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>LSApplicationCategoryType</key><string>public.app-category.productivity</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>Nexus. Everything runs on your Mac.</string>
  <key>NSCameraUsageDescription</key><string>Nexus tracks your hand to move the pointer. Frames never leave this Mac.</string>
  <key>NSMicrophoneUsageDescription</key><string>Nexus listens for "Hey Jarvis". Audio never leaves this Mac.</string>
  <key>NSSpeechRecognitionUsageDescription</key><string>Nexus turns what you say into text, on this Mac.</string>
  <key>NSAppleEventsUsageDescription</key><string>Jarvis controls apps for you, like playing music or making a note.</string>
</dict>
</plist>
PLIST
$PY scripts/make_icon.py "$RES/Nexus.icns" >/dev/null

echo "==> Checking the app can load everything it needs (and doesn't write into itself)"
BEFORE="$(find "$APP" -type f | wc -l)"
env -i HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 "$EXE" -c '
import os, sys, site
site.addsitedir(os.path.join(os.path.dirname(sys.prefix), "site-packages"))  # prefix = Resources/python
import nexus, nexus.app, mediapipe, cv2, faster_whisper, openwakeword, onnxruntime, sounddevice, AppKit, Quartz
print("    ok: Python", sys.version.split()[0], "| nexus", nexus.__version__)
' || { echo "The bundled app failed to import its packages; see the error above."; exit 1; }
[ "$(find "$APP" -type f | wc -l)" = "$BEFORE" ] || { echo "The app wrote files into its own bundle."; exit 1; }

echo "==> Signing"
if [ -n "${NEXUS_SIGN_ID:-}" ]; then
  # Every binary inside is signed with your Developer ID and the hardened runtime, then the app
  find "$APP" -type f \( -name '*.so' -o -name '*.dylib' \) -print0 | \
    xargs -0 -r codesign --force --timestamp --options runtime --sign "$NEXUS_SIGN_ID"
  find "$RES/python/bin" -type f -perm -u+x -print0 | \
    xargs -0 -r codesign --force --timestamp --options runtime --sign "$NEXUS_SIGN_ID"
  codesign --force --timestamp --options runtime --entitlements scripts/entitlements.plist \
    --identifier dev.parnesh.nexus --sign "$NEXUS_SIGN_ID" "$APP"
  codesign --verify --deep --strict "$APP" && echo "    signed: $NEXUS_SIGN_ID"
else
  codesign --force --deep --sign - --identifier dev.parnesh.nexus "$APP" >/dev/null
  echo "    ad-hoc signed (runs on this Mac; other Macs need a Developer ID signature, see the top of this script)"
fi

echo "==> Disk image"
STAGE="$DIST/dmg"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
DMG="$DIST/Nexus-$VERSION.dmg"
hdiutil create -volname "Nexus $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"
if [ -n "${NEXUS_SIGN_ID:-}" ]; then
  codesign --force --timestamp --sign "$NEXUS_SIGN_ID" "$DMG"
fi

if [ -n "${NEXUS_NOTARY_PROFILE:-}" ] && [ -n "${NEXUS_SIGN_ID:-}" ]; then
  echo "==> Notarizing (a few minutes)"
  xcrun notarytool submit "$DMG" --keychain-profile "$NEXUS_NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
  xcrun stapler staple "$APP"
fi

echo ""
echo "Done:"
echo "  $APP  ($(du -sh "$APP" | cut -f1))"
echo "  $DMG  ($(du -sh "$DMG" | cut -f1))"
