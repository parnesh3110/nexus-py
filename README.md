# NEXUS (Python)

Control your Mac with your hand and your voice. Point with your palm, pinch to click, say "Hey Jarvis" and ask for things. Everything runs locally: no cloud AI, nothing leaves the Mac.

**Status: Phases 1-6 built.** Hand cursor, local voice, intent matching, local LLM planner, dictation, a PyTorch gesture model with a rules baseline, and a benchmark suite. Results tables go in `results/` once recorded on this Mac.

## Run it

```bash
bash scripts/setup.sh              # once: Python 3.11 env, packages, all local models (includes Ollama + Qwen 2.5 3B)
bash scripts/make_app.sh           # builds Nexus.app in /Applications (add --login to open it at login)
```

Then open **Nexus** from Spotlight or Launchpad. The first time, a setup guide walks you through permissions and a quick test; after that, opening Nexus shows its window (status, on/off, how to use it). The menu bar icon has the same switches. Your choices are saved in `~/.config/nexus/settings.json`.

**Updates are automatic.** Nexus.app is a small stub that starts `scripts/launch.sh` from this folder. When the code changes, the running app waits until you're not using it and restarts itself; when `requirements.txt` changes, the next start installs the new packages. Rebuild the app only if you move this folder.

From Terminal instead (with flags, for development):

```bash
source .venv/bin/activate
python -m nexus --preview          # hand cursor + Jarvis, with a camera window showing both
```

macOS asks for **Camera**, **Microphone** and **Accessibility** for your terminal, and "Terminal wants to control …" the first time Jarvis scripts an app.

| Flag | What it does |
|------|--------------|
| `--preview` | camera window: hand skeleton, pinch ratio, fps, and what Jarvis heard / said |
| `--no-hands` | voice only, the camera never turns on |
| `--no-voice` | hand cursor only, the mic never turns on |
| `--record` | every frame + all 21 landmarks to `runs/` (benchmarks and ML training data) |
| `--whisper tiny.en/base.en/small.en` | speech-to-text speed vs. accuracy |
| `--llm qwen2.5:7b` | a bigger local model for complex requests |
| `--voice Nathan` | speaking voice |
| `--sensitivity low/medium/high/max` | hand movement needed to cross the screen |

## Calibrate your hand

Pinch and the Jarvis panel gesture work best once Nexus knows your hand: menu bar icon or the Nexus window, **Calibrate Hand…** (⌘K). Four poses, about 2 seconds each: open, pinch, all fingertips together, spread. Nexus measures your own levels and places the pinch and panel thresholds between them, so a light pinch counts in the positions you actually use. Clicks and the panel pause while it runs. Every 5 s the log records your real pinch and spread ranges (`hand.stats`), which is what the thresholds are tuned from.

## Release build (a Nexus.app you can share)

```bash
bash scripts/build_release.sh
```

Builds `dist/Nexus.app` and `dist/Nexus-<version>.dmg`: Python, packages, code and models all inside the app, nothing from this folder, no Homebrew, no Terminal. Intent matching runs on fastembed (ONNX), so PyTorch stays out of the app (the optional trained gesture model needs PyTorch, so release builds skip it). Your files go where Mac apps keep them: settings, recordings and results in `~/Library/Application Support/Nexus`, logs in `~/Library/Logs/Nexus`. The app never writes into its own bundle (everything is precompiled), so its signature stays valid.

To publish it for other Macs without a Gatekeeper warning you need a paid Apple Developer account:

```bash
xcrun notarytool store-credentials nexus --apple-id you@example.com --team-id TEAMID   # once
NEXUS_SIGN_ID="Developer ID Application: Your Name (TEAMID)" NEXUS_NOTARY_PROFILE=nexus bash scripts/build_release.sh
```

That signs every binary inside with the hardened runtime and `scripts/entitlements.plist` (camera, microphone, app control), notarizes the disk image with Apple and staples the ticket. The local AI (Ollama) isn't bundled: the setup guide opens its download page, then downloads the model for you.

## Hand control

```
camera thread (OpenCV, 640x480, newest frame only)
  -> MediaPipe Hand Landmarker, VIDEO mode (21 landmarks + 3D world landmarks)
  -> palm point (center of the 4 knuckles) -> One Euro filter -> sliding camera box -> screen
  -> rubber band + still-lock -> pinch (3D thumb-index gap / hand size, hysteresis) -> click / drag
  -> 120 Hz cursor animator (prediction only on fast moves) -> Quartz mouse events
```

- **Knuckle-center palm point, not the fingertip:** in the earlier Swift version the fingertip jumped ~138 px per pinch, the palm point ~8 px.
- **3D pinch:** a flat 2D distance shrinks when the hand tilts toward the camera and fakes a click.
- **Clicks snap back to where you aimed** before the fingers started closing.
- **Click vs. drag:** a drag needs 40 px after 0.25 s, or 150 px at once. Without this, 18 of 44 quick clicks became drags.
- **Ghost hands and glitches ignored:** a new hand must show up two frames in a row; palm jumps faster than 2.5 camera-widths/s are dropped.

## Jarvis

Say **"Hey Jarvis"** (pop = listening, tink = got it), then your request.

```
mic 16 kHz -> openWakeWord "hey jarvis" (small neural net, every 80 ms)
  -> record until you pause -> faster-whisper base.en (int8, CPU, primed with your app names)
  -> understand, fastest first:
       1. rules    exact phrases, "open X" / "play X", multi-step with and/then           ~0 ms
       2. intents  sentence-transformers MiniLM: rephrasings -> known commands           ~10 ms
       3. planner  Qwen 2.5 3B via Ollama, JSON-schema output: plans steps, writes text,
                   uses AppleScript, finds files, looks at results and plans the rest   ~2-4 s
  -> actions: apps, windows, menus, buttons, keys, typing, music, web, volume, AppleScript, shell, files
  -> reply with macOS `say` (best installed voice)
```

Things to try:

| You say | What happens | Layer |
|---|---|---|
| "open Apple Music and play The Reason" | opens Music, plays the song | rules |
| "shrink this window" / "it's too loud" | minimize / volume down | intents |
| "open Notes and write a shopping list with milk, eggs and bread" | new note, the LLM writes the list and types it | planner |
| "turn on dark mode" | AppleScript | planner |
| "open my resume" | Spotlight search, then opens the newest match | planner (2 rounds) |
| "open Word and draft what I'm saying" | new Word document, then dictation | rules |
| "Hey Jarvis, stay" | conversation mode until "that's all" | |

**Dictation:** everything you say is typed with punctuation. "Scratch that" deletes the last part, "new line" / "new paragraph", "replace Monday with Tuesday" (the local LLM rewrites, only the changed characters are retyped), "stop dictation" ends it.

**Safety:** AppleScript that deletes, sends, erases or restarts, and any shell command that isn't read-only, is asked out loud first: "Should I …?" Silence counts as no.

## Jarvis panel

A Siri-style panel (frosted glass, animated orb, the ongoing conversation), controlled by your fingers:

- **Open:** bring all five fingertips together, then spread them. The panel opens as far as your fingers do, and once they're fully spread it stays open and Jarvis starts listening.
- **Close:** spread your hand, then bring the fingertips together. 80% closed fingers = an 80% closed panel. Stop halfway and relax, and it settles to the nearer end.
- **Move:** pinch on the panel and drag.

Opening has to start from bunched fingertips, which you don't do while pointing, so it doesn't pop open by accident. Bunching all five fingers is not counted as a click. With `--no-hands` the panel shows itself while Jarvis is busy.

**Why the camera and not another sensor:** MacBooks have no depth, LiDAR or IR sensor, and the trackpad only sees fingers touching it. Instead, when no hand is in view, a 64x48 thumbnail diff runs first and MediaPipe only runs when something moves (plus a check every 15th frame), which makes idle tracking nearly free.

## Gesture model (PyTorch)

A learned alternative to hand-written gesture rules, trained on your own recordings.

```bash
python -m nexus.collect     # ~5 min: 25 of each gesture on cue + 60 s of free movement. Do 2-3 sessions.
python -m nexus.train       # trains, tests on a held-out session, compares against the rules
python -m nexus --gestures  # live: swipe left/right = switch desktop, open palm = stop
```

- **Classes:** idle, pinch (tap), swipe left, swipe right, open palm.
- **Input:** 30-frame (~1 s) windows of 68 features per frame: 3D world landmarks relative to the wrist and scaled by hand size (hand shape), palm path and velocity in hand sizes (motion), pinch ratio.
- **Model:** 1D temporal CNN, 3 conv layers with dilation 1/2/4, mean + max pooling over time, linear head. ~55k parameters, well under 1 ms per prediction on the CPU.
- **Training:** AdamW, cosine schedule, class-weighted cross-entropy with label smoothing; augmentation with noise, time shifts and path scaling; best checkpoint by validation accuracy.
- **Evaluation:** with 2+ sessions the newest is held out entirely. The rule baseline runs on the exact same test clips. Both run over the free-movement streams, which are never used in training, with the live app's event rules (threshold, 3-frame confirmation, cooldown) to get **false activations per minute**. Output: `results/gesture-<time>.md`.

Clicks and drags stay on the rule-based detector: they need a per-frame answer, not a 1-second window.

## Benchmarks

```bash
python -m nexus.bench                      # hand tests, ~2 min
python -m nexus.bench --voice --gestures   # + voice tests and false gestures with the trained model
```

A window tells you what to do: hold still (jitter), pinch on 20 cues (click detection rate), move without pinching (false clicks/min), hand down (idle CPU). Throughout: fps, frame-to-pointer latency, MediaPipe time, CPU, RAM. Voice: wake-word hit rate, speech-to-text word error rate, transcription time, and which layer understood each command (nothing is executed). Output: `results/bench-<time>.md`.

## Logs and measurements

Every run writes `logs/nexus-<time>.log` (JSON lines). `perf.summary` every 5 s: fps, render fps, latency p50/p95, MediaPipe time, CPU %, RAM. State changes are logged as they happen: hand acquired/lost, pinch down/up, wake word score, what Whisper heard and how long it took, which layer understood it (`jarvis.understood`, `nlu.match` with similarity, `llm.plan` with timing and steps), every action and its result. `NEXUS_LOG=debug` prints debug events too.

## Tests

```bash
python -m pytest -q
```

38 tests, no camera, mic or models needed: pointer stability, click vs. drag, ghost hands, glitch rejection; command parsing including sentences Whisper got wrong in real logs; which scripts and shell commands need confirmation; LLM plan validation (unknown actions and invented arguments are dropped, never crash); dictation spacing and spoken punctuation; gesture features (translation invariance, gap filling), the rule baseline, and a check that the PyTorch model learns synthetic gestures.

## Roadmap

- [x] Phase 1: MediaPipe hand tracking, stabilized cursor, click/drag, logging, tests
- [x] Phase 2: "Hey Jarvis" wake word + local Whisper speech-to-text
- [x] Phase 3: intent matching with sentence-transformers; actions (apps, windows, menus, buttons, AppleScript, shell, files)
- [x] Phase 4: local LLM planner (Ollama) with multi-round planning; dictation with corrections; spoken confirmations
- [x] Phase 5: PyTorch temporal gesture classifier trained on recorded landmarks, vs. the rule-based baseline
- [x] Phase 6: benchmark suite: gesture accuracy, false activations, latency, fps, CPU, RAM, wake rate, WER
- [x] Phase 7: Siri-style panel, menu bar app (Nexus.app)
