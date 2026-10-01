#!/usr/bin/env bash
# Builds Nexus.app. Its executable is a small compiled launcher (scripts/nexus_stub.c) that runs
# the project's Python inside its own process, so macOS shows "Nexus" in Privacy & Security, the
# Dock, the app switcher and Activity Monitor. The app keeps itself up to date (nexus/bootstrap.py):
# code and dependency changes apply automatically. Rebuild only if you move the project folder.
#
#   bash scripts/make_app.sh           # -> /Applications/Nexus.app (or ~/Applications)
#   bash scripts/make_app.sh --login   # also open it automatically when you log in
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [ ! -x .venv/bin/python ]; then
  echo "Run bash scripts/setup.sh first."
  exit 1
fi

DEST=/Applications
[ -w "$DEST" ] || { DEST="$HOME/Applications"; mkdir -p "$DEST"; }
APP="$DEST/Nexus.app"
EXE="$APP/Contents/MacOS/Nexus"
echo "==> Building $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# --- the executable: compiled launcher (preferred) ---
PYVER="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
BASE="$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')"
INC="$BASE/include/python$PYVER"
LIB="$BASE/lib/libpython$PYVER.dylib"
KIND="script"
if command -v clang >/dev/null 2>&1 && [ -f "$INC/Python.h" ] && [ -f "$LIB" ]; then
  echo "==> Compiling the launcher against Python $PYVER ($BASE)"
  if clang -O2 -Wall -o "$EXE" scripts/nexus_stub.c -I"$INC" "$LIB" -Wl,-rpath,"$BASE/lib" \
       -DNEXUS_ROOT="\"$ROOT\"" -DNEXUS_PYHOME="\"$BASE\""; then
    # The library may record a build-machine location for itself: point the launcher at the real file
    OLD="$(otool -L "$EXE" | awk '/libpython/ {print $1; exit}')"
    if [ -n "$OLD" ] && [[ "$OLD" != @rpath/* ]] && [ ! -e "$OLD" ]; then
      install_name_tool -change "$OLD" "$LIB" "$EXE"
    fi
    # Prove it works before trusting it: run it as an interpreter
    if CHECK="$("$EXE" -c 'import sys, nexus; print(sys.version.split()[0])' 2>&1)"; then
      echo "    launcher works (Python $CHECK)"
      KIND="compiled"
    else
      echo "    launcher didn't run ($CHECK), using the script version instead"
    fi
  fi
else
  echo "==> No compiler or Python headers found, using the script version"
fi

if [ "$KIND" = "script" ]; then
  # Fallback: works, but macOS will list "python3.11" for permissions and it won't be in the Dock
  cat > "$EXE" <<LAUNCHER
#!/bin/bash
LAUNCH="$ROOT/scripts/launch.sh"
if [ ! -f "\$LAUNCH" ]; then
  osascript -e 'display dialog "Nexus can not find its project folder ($ROOT). Was it moved?" buttons {"OK"} default button 1 with title "Nexus"'
  exit 1
fi
/bin/bash "\$LAUNCH" "$APP"
LAUNCHER
  chmod +x "$EXE" scripts/launch.sh
fi

# Compiled: a normal app (Dock, app switcher). Script: menu bar only (a script can't own a Dock icon).
UI_ELEMENT="<false/>"
[ "$KIND" = "script" ] && UI_ELEMENT="<true/>"
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
  <key>CFBundleShortVersionString</key><string>0.8</string>
  <key>CFBundleVersion</key><string>8</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSUIElement</key>$UI_ELEMENT
  <key>NSCameraUsageDescription</key><string>Nexus tracks your hand to move the pointer. Frames never leave this Mac.</string>
  <key>NSMicrophoneUsageDescription</key><string>Nexus listens for "Hey Jarvis". Audio never leaves this Mac.</string>
  <key>NSSpeechRecognitionUsageDescription</key><string>Nexus turns what you say into text, on this Mac.</string>
  <key>NSAppleEventsUsageDescription</key><string>Jarvis controls apps for you, like playing music or making a note.</string>
</dict>
</plist>
PLIST

echo "==> Drawing the icon"
.venv/bin/python scripts/make_icon.py "$APP/Contents/Resources/Nexus.icns"

# A local signature with a fixed identifier: macOS remembers Nexus's permissions across rebuilds
codesign --force --sign - --identifier dev.parnesh.nexus "$APP" >/dev/null 2>&1 || echo "(codesign skipped)"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" >/dev/null 2>&1 || true

if [ "${1:-}" = "--login" ]; then
  osascript -e "tell application \"System Events\" to make login item at end with properties {path:\"$APP\", hidden:true}" >/dev/null
  echo "==> Nexus will open when you log in (remove it in System Settings -> General -> Login Items)"
fi

echo ""
echo "Done: $APP ($KIND launcher)"
if [ "$KIND" = "compiled" ]; then
  echo "Nexus now shows up as \"Nexus\" in Privacy & Security, the Dock and the app switcher."
  echo "macOS will ask for Camera, Microphone and Accessibility again, this time for Nexus."
  echo "You can remove the old \"python3.11\" entries in System Settings -> Privacy & Security."
fi
echo "Open it from Spotlight (cmd+space, \"Nexus\"), Launchpad or the Dock."
