"""Guided benchmark: every number in the README and on the resume comes from here.

    python -m nexus.bench                 # hand tests, ~2 minutes
    python -m nexus.bench --voice         # + voice tests (say 8 prompted commands)
    python -m nexus.bench --gestures      # + false gestures/min with the trained model

Hand tests (a window tells you what to do):
  1. still     10 s   hold your hand still        -> pointer jitter (px RMS), raw vs. stabilized
  2. clicks    20x    pinch when it says PINCH    -> click detection rate, cue -> click time
  3. free      40 s   move around, don't pinch    -> false clicks / min (+ false gestures / min)
  4. no hand   15 s   hand down                   -> idle CPU
  Throughout: fps, frame -> pointer latency, MediaPipe time, CPU, RAM.
Voice tests: wake-word hit rate, speech-to-text word error rate, transcription time, and
which layer understood each command. Nothing is executed.
Results: results/bench-<time>.json and .md
"""
import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

from .collect import GREEN, WHITE, YELLOW, Session
from .log import log
from .mouse import screen_size
from .pointer import PointerEngine

VOICE_PROMPTS = ["open Safari", "minimize", "new tab", "play The Reason", "volume down",
                 "open Apple Music and play The Reason", "shrink this window", "go to github"]


def _pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def wer(ref: str, hyp: str) -> float:
    """Word error rate: word-level edit distance / words in the reference."""
    norm = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


class Phase:
    """Runs frames through the pointer engine for a while, collecting everything."""

    def __init__(self, s: Session, engine: PointerEngine, recognizer=None):
        self.s, self.engine, self.rec = s, engine, recognizer
        self.proc = psutil.Process()

    def run(self, seconds: float, lines: list[str], color=GREEN, on_frame=None) -> dict:
        self.proc.cpu_percent(None)
        out = {"latency": [], "vision": [], "x": [], "y": [], "rx": [], "ry": [], "hand": [], "downs": [], "gestures": [],
               "frames": 0}
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            bgr, hand, t, vision_ms = self.s.frame()
            st = self.engine.update(hand, t)
            if self.rec is not None:
                g = self.rec.update(hand, t)
                if g:
                    out["gestures"].append((t, g))
            out["latency"].append((time.perf_counter() - t) * 1000)
            out["vision"].append(vision_ms)
            out["hand"].append(st.visible)
            if st.visible:
                out["x"].append(st.x); out["y"].append(st.y); out["rx"].append(st.raw_x); out["ry"].append(st.raw_y)
            if st.event == "down":
                out["downs"].append(t)
            out["frames"] += 1
            extra = on_frame(t) if on_frame else None
            text = lines + ([extra] if extra else [])
            if self.s.show(bgr, hand, text, color, (time.perf_counter() - start) / seconds) == ord("q"):
                raise KeyboardInterrupt
        out["seconds"] = time.perf_counter() - start
        out["cpu_pct"] = self.proc.cpu_percent(None)
        out["rss_mb"] = self.proc.memory_info().rss / 1_048_576
        return out

    def countdown(self, lines, seconds=3.0):
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            bgr, hand, _, _ = self.s.frame()
            self.s.show(bgr, hand, lines + [f"starting in {seconds - (time.perf_counter() - start):.0f}"], YELLOW)


def jitter(xs, ys, window=15) -> float:
    """Median RMS spread over 0.5 s windows (how much a still pointer wobbles)."""
    xs, ys = np.array(xs), np.array(ys)
    vals = []
    for i in range(0, len(xs) - window, 5):
        x, y = xs[i:i + window], ys[i:i + window]
        vals.append(math.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2)))
    return float(np.median(vals)) if vals else float("nan")


def hand_tests(s: Session, recognizer) -> dict:
    W, H = screen_size()
    engine = PointerEngine(W, H, s.cfg, s.cam.aspect)
    ph = Phase(s, engine, recognizer)
    res = {}

    ph.countdown(["TEST 1/4: HOLD STILL", "Raise your hand and keep it as still as you comfortably can."])
    still = ph.run(10, ["HOLD STILL"])
    res["still"] = {"jitter_raw_px": jitter(still["rx"], still["ry"]), "jitter_px": jitter(still["x"], still["y"]),
                    "hand_pct": 100 * float(np.mean(still["hand"]))}

    ph.countdown(["TEST 2/4: CLICKS", "Pinch once each time it says PINCH (20 times)."])
    cues, gap, n = [], 2.5, 20
    t_start = time.perf_counter()
    def cue_text(t):
        k = int((t - t_start) // gap)
        if k >= n:
            return None
        if not cues or cues[-1] < t_start + k * gap:
            cues.append(t_start + k * gap)
        return "PINCH NOW" if (t - t_start) % gap < 0.8 else f"({k + 1}/{n}) get ready"
    clicks = ph.run(gap * n + 1.0, ["CLICKS"], GREEN, cue_text)
    hits, times, used = 0, [], set()
    for c in cues:
        match = next((d for d in clicks["downs"] if c <= d <= c + 1.5 and d not in used), None)
        if match is not None:
            hits += 1
            used.add(match)
            times.append((match - c) * 1000)
    res["clicks"] = {"cues": len(cues), "detected": hits, "detection_rate": hits / max(len(cues), 1),
                     "extra_clicks": len(clicks["downs"]) - len(used), "cue_to_click_ms_p50": _pct(times, 50)}

    ph.countdown(["TEST 3/4: FREE MOVEMENT", "Move your hand around and point at things. Don't pinch."])
    free = ph.run(40, ["FREE MOVEMENT: no pinching"])
    minutes = free["seconds"] / 60
    res["free"] = {"minutes": minutes, "false_clicks_per_min": len(free["downs"]) / minutes,
                   "false_gestures_per_min": (len(free["gestures"]) / minutes) if recognizer else None,
                   "false_gestures": [g for _, g in free["gestures"]]}

    ph.countdown(["TEST 4/4: NO HAND", "Put your hand down, out of view."])
    idle = ph.run(15, ["NO HAND: keep it down"])
    res["idle"] = {"cpu_pct": idle["cpu_pct"], "note": "benchmark processes every frame; the app's idle mode skips 2 of 3"}

    active = still["latency"] + clicks["latency"] + free["latency"]
    vision = still["vision"] + clicks["vision"] + free["vision"]
    secs = still["seconds"] + clicks["seconds"] + free["seconds"]
    res["perf"] = {"fps": (still["frames"] + clicks["frames"] + free["frames"]) / secs,
                   "latency_p50_ms": _pct(active, 50), "latency_p95_ms": _pct(active, 95),
                   "mediapipe_p50_ms": _pct(vision, 50), "mediapipe_p95_ms": _pct(vision, 95),
                   "cpu_pct_active": float(np.mean([still["cpu_pct"], clicks["cpu_pct"], free["cpu_pct"]])),
                   "rss_mb": max(still["rss_mb"], free["rss_mb"])}
    return res


def voice_tests(s: Session) -> dict:
    from .commands import parse, strip_fillers
    from .intents import IntentRouter
    from .voice import Mic, Transcriber, WakeWord, record_utterance
    stt, wake, mic, router = Transcriber(), WakeWord(), Mic(), IntentRouter()
    router.ready.wait(60)
    mic.start()
    rows = []
    blank = np.zeros((480, 640, 3), np.uint8)
    for i, phrase in enumerate(VOICE_PROMPTS):
        mic.drain()
        wake.reset()
        s.show(blank, None, [f"VOICE {i + 1}/{len(VOICE_PROMPTS)}", f'Say: "Hey Jarvis, {phrase}"'], YELLOW)
        start, woke = time.perf_counter(), False
        while time.perf_counter() - start < 10:
            s.show(blank, None, [f"VOICE {i + 1}/{len(VOICE_PROMPTS)}", f'Say: "Hey Jarvis, {phrase}"'], YELLOW)
            if wake.score(mic.read(1.0)) >= wake.threshold:
                woke = True
                break
        row = {"prompt": phrase, "wake": woke}
        if woke:
            s.show(blank, None, ["LISTENING", f'"{phrase}"'], GREEN)
            audio = record_utterance(mic, start_timeout=4.0)
            if audio is not None:
                t0 = time.perf_counter()
                text = stt(audio)
                row.update(heard=text, stt_ms=(time.perf_counter() - t0) * 1000, wer=wer(phrase, strip_fillers(text)))
                steps = parse(text)
                row["layer"] = "rules" if steps else ("intent" if router.route(strip_fillers(text)) else "llm")
        rows.append(row)
        log.info("bench", "voice", **row)
    mic.stop()
    heard = [r for r in rows if "wer" in r]
    return {"prompts": len(rows), "wake_rate": sum(r["wake"] for r in rows) / len(rows),
            "wer_mean": float(np.mean([r["wer"] for r in heard])) if heard else None,
            "stt_ms_p50": _pct([r["stt_ms"] for r in heard], 50),
            "layers": {k: sum(r.get("layer") == k for r in rows) for k in ("rules", "intent", "llm")},
            "rows": rows}


def to_markdown(r: dict, stamp: str) -> str:
    L = [f"# NEXUS benchmark ({stamp})", "", f"{r['system']['chip']}, {r['system']['camera']}", "", "| Metric | Result |",
         "|---|---|"]
    h = r.get("hand")
    if h:
        L += [f"| Still-hand jitter (raw -> stabilized) | {h['still']['jitter_raw_px']:.1f} -> {h['still']['jitter_px']:.1f} px |",
              f"| Click detection | {h['clicks']['detected']}/{h['clicks']['cues']} ({h['clicks']['detection_rate']:.0%}) |",
              f"| Cue -> click (p50, includes your reaction) | {h['clicks']['cue_to_click_ms_p50']:.0f} ms |",
              f"| False clicks, free movement | {h['free']['false_clicks_per_min']:.2f} / min |"]
        if h["free"]["false_gestures_per_min"] is not None:
            L.append(f"| False gestures (model), free movement | {h['free']['false_gestures_per_min']:.2f} / min |")
        p = h["perf"]
        L += [f"| Tracking rate | {p['fps']:.1f} fps |",
              f"| Frame -> pointer latency p50 / p95 | {p['latency_p50_ms']:.1f} / {p['latency_p95_ms']:.1f} ms |",
              f"| MediaPipe per frame p50 / p95 | {p['mediapipe_p50_ms']:.1f} / {p['mediapipe_p95_ms']:.1f} ms |",
              f"| CPU active / no hand | {p['cpu_pct_active']:.0f}% / {h['idle']['cpu_pct']:.0f}% |",
              f"| Memory | {p['rss_mb']:.0f} MB |"]
    v = r.get("voice")
    if v:
        L += [f"| Wake word hit rate | {v['wake_rate']:.0%} ({v['prompts']} prompts) |",
              f"| Speech-to-text word error rate | {v['wer_mean']:.1%} |" if v["wer_mean"] is not None else "| WER | - |",
              f"| Transcription time p50 | {v['stt_ms_p50']:.0f} ms |",
              f"| Understood by rules / intents / LLM | {v['layers']['rules']} / {v['layers']['intent']} / {v['layers']['llm']} |"]
    L.append("")
    L.append("CPU % is of one core (100% = one full core). Latency is from the frame arriving to the pointer update.")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nexus.bench")
    ap.add_argument("--voice", action="store_true", help="also run the voice tests")
    ap.add_argument("--no-hands", action="store_true", help="skip the hand tests")
    ap.add_argument("--gestures", default=None, nargs="?", const="models/gesture.pt",
                    help="measure false gestures with the trained model")
    args = ap.parse_args(argv)
    log.open()
    from .perf import system_info
    s = Session("NEXUS benchmark")
    results = {"system": {**system_info(), "camera": f"{s.cam.width}x{s.cam.height}"}}
    try:
        if not args.no_hands:
            recognizer = None
            if args.gestures:
                from .gesture_model import GestureRecognizer
                recognizer = GestureRecognizer(args.gestures)
            results["hand"] = hand_tests(s, recognizer)
        if args.voice:
            results["voice"] = voice_tests(s)
    except KeyboardInterrupt:
        print("stopped early; saving what was measured")
    finally:
        s.close()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/bench-{stamp}.json").write_text(json.dumps(results, indent=2, default=float))
    md = to_markdown(results, stamp)
    Path(f"results/bench-{stamp}.md").write_text(md)
    log.info("bench", "done", file=f"results/bench-{stamp}.md")
    print("\n" + md)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
