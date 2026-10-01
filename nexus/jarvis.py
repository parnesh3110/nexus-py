"""Jarvis: "Hey Jarvis" -> listen -> transcribe -> understand -> do it -> answer. Own thread.

Understanding, fastest first, all on this Mac:
  1. rules      exact phrases and "open X" / "play X" patterns, multi-step with and/then   (~0 ms)
  2. intents    sentence-transformers: rephrasings ("shrink this window" -> minimize)      (~10 ms)
  3. planner    local LLM (Ollama): anything else, writes text, AppleScript, files; can
                look at results and plan the rest                                           (~2-4 s)

Risky scripts/commands are confirmed out loud first. "Draft what I'm saying" starts dictation.
Conversation mode: "Hey Jarvis, stay" keeps listening between turns; "that's all" ends it.
"""
import queue
import threading
import time
from collections import deque

from . import actions
from .apps import index
from .commands import Step, join_dictation, parse, strip_fillers
from .log import log
from .safety import is_yes, script_needs_confirm, shell_needs_confirm
from .speak import Speaker, chime
from .voice import Mic, Transcriber, WakeWord, record_utterance

STAY = {"stay", "stay with me", "let's talk", "lets talk", "keep listening", "conversation mode", "talk to me"}
LEAVE = {"that's all", "thats all", "that's it", "goodbye", "bye", "stop listening", "we're done", "exit", "go to sleep"}
CANCEL = {"stop", "cancel", "never mind", "nevermind", "nothing"}

STOP_DICTATION = {"stop dictation", "stop dictating", "end dictation", "done dictating", "i'm done", "im done",
                  "that's all", "thats all", "stop typing", "finish dictation"}
SCRATCH = {"scratch that", "delete that", "undo that", "remove that", "erase that"}
CORRECTION_STARTS = ("replace ", "change ", "no i meant", "no, i meant", "i meant ", "correction", "instead of ",
                     "make that ", "capitalize ", "fix ")
WHISPER_JUNK = {"thanks for watching", "thank you for watching", "you", "[blank_audio]"}  # silence hallucinations
DOC_APPS = {"microsoft word", "pages", "textedit", "notes"}  # need a new document before dictating


def _ms(s: float) -> int:
    return int(round(s * 1000))


class Jarvis(threading.Thread):
    def __init__(self, whisper_model: str = "base.en", voice: str | None = None, wake_threshold: float = 0.35,
                 llm_model: str | None = None, mic: str | None = None):
        super().__init__(name="nexus-jarvis", daemon=True)
        self.whisper_model, self.voice_name, self.wake_threshold = whisper_model, voice, wake_threshold
        self.llm_model, self.mic_name = llm_model, mic
        self.state = "starting"
        self.heard = ""   # last thing you said, for the UI
        self.reply = ""   # last thing Jarvis said / typed
        self.staying = False
        self.dictating = False
        self._dictate_next = False
        self._force_turn = False
        self.conversation: deque = deque(maxlen=20)  # ("you" | "jarvis", text), for the panel
        # The panel holds the conversation open: while it's open Jarvis keeps listening, no wake word
        self.hold_open = False
        self._cancel_listen = False
        self.inbox: queue.Queue = queue.Queue()  # messages typed in the panel
        # For the setup guide's live checklist and meter
        self.last_score = 0.0
        self.wake_count = 0
        self.turns = 0
        self.near_misses = 0
        self._running = True
        self.intents = None
        self.planner = None
        self._planner_status = (False, "starting")
        self._planner_checked = 0.0

    # --- lifecycle ---

    def _set(self, state: str, reply: str | None = None):
        if state != self.state:
            log.debug("jarvis", "state", frm=self.state, to=state)
        self.state = state
        if reply is not None:
            self.reply = reply

    def stop(self):
        self._running = False

    def request_listen(self):
        """Start listening right away, no wake word (the panel was just opened). Thread-safe."""
        if self.state == "idle" and not self.paused:
            self._force_turn = True

    def set_hold_open(self, on: bool):
        """The Jarvis panel opened (on) or closed (off). While it's open Jarvis is in a conversation:
        it listens again after every reply, and silence doesn't end it. Thread-safe."""
        if on == self.hold_open:
            return
        self.hold_open = on
        log.info("jarvis", "conversation_on" if on else "conversation_off", reason="panel")
        if on:
            self.staying = True
            self.request_listen()
        else:
            self.staying = False
            self._cancel_listen = True  # stop the current listen right away

    def submit_text(self, text: str):
        """A message typed in the panel. Handled like something you said. Thread-safe."""
        text = text.strip()
        if text:
            self.inbox.put(text)
            log.info("jarvis", "typed", chars=len(text))

    def _typed(self) -> str | None:
        try:
            return self.inbox.get_nowait()
        except queue.Empty:
            return None

    paused = False

    def pause(self):
        """Menu bar: Jarvis off. The mic stream is closed, so macOS's orange mic dot goes away."""
        if self.paused:
            return
        self.paused = True
        if getattr(self, "speaker", None):
            self.speaker.stop()
        if getattr(self, "mic", None):
            self.mic.stop()
        self._set("off")
        log.info("jarvis", "paused")

    def resume(self):
        if not self.paused:
            return
        if getattr(self, "mic", None):
            self.mic.start(self.mic_name)
        self.paused = False
        self._set("idle")
        log.info("jarvis", "resumed")

    def run(self):
        try:
            index.load()
            self.speaker = Speaker(self.voice_name)
            self.stt = Transcriber(self.whisper_model)
            self.wake = WakeWord(self.wake_threshold)
            self.mic = Mic()
            if not self.paused:  # first launch: the mic waits until the setup guide has explained it
                self.mic.start(self.mic_name)
        except Exception as e:
            log.error("jarvis", "startup_failed", error=str(e),
                      fix="run scripts/setup.sh, and allow Microphone for your terminal")
            self.state = "off"
            return
        # The smarter layers load in the background; simple commands work right away
        from .intents import IntentRouter
        from .planner import Planner
        self.intents = IntentRouter()
        self.planner = Planner(self.llm_model) if self.llm_model else Planner()
        threading.Thread(target=self._warm_planner, name="nexus-llm-warm", daemon=True).start()
        if index.ready.wait(3):
            self.stt.set_hints(index.names())  # so Whisper spells your app names right
        if self.paused:  # switched off in the menu while the models were loading
            self.mic.stop()
            self._set("off")
        else:
            self._set("idle")
        log.info("jarvis", "ready", say="Hey Jarvis", paused=self.paused)

        cooldown = 0.0
        peak, peak_t = 0.0, time.perf_counter()
        while self._running:
            if not self.inbox.empty():  # typed in the panel: works even with the mic off
                try:
                    self._turn(listen=not self.paused)
                except Exception as e:
                    log.error("jarvis", "turn_failed", error=repr(e))
                    self._set("off" if self.paused else "idle")
                continue
            if self.paused:
                time.sleep(0.2)
                continue
            try:
                chunk = self.mic.read(0.3)
            except queue.Empty:
                continue
            forced = self._force_turn
            self._force_turn = False
            score = self.wake.score(chunk)
            self.last_score = max(score, self.last_score * 0.85)  # decaying peak, for the setup meter
            # Near misses: something close to "Hey Jarvis" that didn't reach the threshold. If these
            # show up when you really said it, lower --wake-threshold.
            if score > peak:
                peak = score
            if time.perf_counter() - peak_t > 3.0:
                if 0.2 <= peak < self.wake.threshold:
                    self.near_misses += 1
                    log.info("voice", "wake_near_miss", peak=peak, threshold=self.wake.threshold)
                peak, peak_t = 0.0, time.perf_counter()
            if forced or (score >= self.wake.threshold and time.perf_counter() > cooldown):
                if not forced:
                    self.wake_count += 1
                log.info("voice", "wake", score=score, while_speaking=self.speaker.speaking, by="panel" if forced else "voice")
                self.speaker.stop()  # talking over Jarvis interrupts it
                try:
                    self._turn()
                except Exception as e:  # never let one bad turn kill the assistant
                    log.error("jarvis", "turn_failed", error=repr(e))
                    self._set("idle")
                self.wake.reset()
                self.mic.drain()
                cooldown = time.perf_counter() + 1.0
        self.mic.stop()

    def _warm_planner(self):
        ok, why = self._planner_ok(force=True)
        if ok:
            self.planner.warm_up()
        else:
            log.warn("llm", "unavailable", reason=why, note="simple commands still work")

    def _planner_ok(self, force: bool = False) -> tuple[bool, str]:
        if force or time.perf_counter() - self._planner_checked > 20:
            self._planner_status = self.planner.available()
            self._planner_checked = time.perf_counter()
        return self._planner_status

    # --- one conversation turn ---

    def _listen_and_transcribe(self, start_timeout: float) -> tuple[str | None, int]:
        self._cancel_listen = False
        typed = self._typed()
        if typed:
            return typed, 0
        audio = record_utterance(self.mic, start_timeout=start_timeout,
                                 should_stop=lambda: self._cancel_listen or not self.inbox.empty())
        typed = self._typed()  # typing wins over half-said speech
        if typed:
            return typed, 0
        if audio is None:
            return None, 0
        self._set("thinking")
        s0 = time.perf_counter()
        text = self.stt(audio)
        stt_ms = _ms(time.perf_counter() - s0)
        log.info("voice", "command", text=text, audio_s=round(len(audio) / 16000, 2), stt_ms=stt_ms)
        return text, stt_ms

    def _speak(self, text: str):
        self._set("speaking", text)
        self.speaker.say(text)
        self.speaker.wait()   # don't listen to ourselves
        self.mic.drain()

    def _turn(self, listen: bool = True):
        """One exchange, or a whole conversation while `staying` (panel open, or "stay").
        listen=False: only answer what was typed (the mic is off)."""
        if listen:
            chime("Pop", 0.7)
        self._set("listening")
        self.heard = ""
        while True:
            t0 = time.perf_counter()
            if listen:
                text, _ = self._listen_and_transcribe(8.0 if self.staying else 5.0)
            else:
                text = self._typed()
            if text is None:
                if listen and self.hold_open:
                    continue  # panel open: silence is fine, keep listening
                if self.staying:
                    self.staying = False
                    log.info("jarvis", "conversation_off", reason="silence")
                if listen:
                    chime("Bottle", 0.3)
                self._set("off" if self.paused else "idle")
                return
            self.heard = text
            if text:
                self.conversation.append(("you", text))
                chime("Tink", 0.4)
                reply = self._handle(text)
                self.turns += 1
                log.info("jarvis", "turn_done", ms=_ms(time.perf_counter() - t0), reply=reply or "")
                if reply:
                    self.conversation.append(("jarvis", reply))
                    self._speak(reply)
            if self._dictate_next and listen:
                self._dictate_next = False
                self._dictate()
            if not listen and self.inbox.empty():
                self._set("off" if self.paused else "idle")
                return
            if listen and not self.staying:
                self._set("idle")
                return
            if listen:
                chime("Pop", 0.3)  # your turn
            self._set("listening")

    # --- understanding ---

    def _handle(self, text: str) -> str | None:
        low = strip_fillers(text).lower().strip(" .!?")
        if low in STAY:
            self.staying = True
            log.info("jarvis", "conversation_on")
            return "I'm listening."
        if self.staying and low in LEAVE:
            self.staying = False
            self.hold_open = False  # "that's all" also ends a panel conversation (the panel stays open)
            log.info("jarvis", "conversation_off", reason="asked")
            return "Okay."
        if low in CANCEL:
            return None

        steps, layer = parse(text), "rules"
        if steps is None and self.intents is not None:
            hit = self.intents.route(strip_fillers(text))
            if hit:
                steps, layer = parse(hit[0]), "intent"
        if steps is not None:
            log.info("jarvis", "understood", layer=layer, steps=[s.action for s in steps])
            ok, reply, _ = self._run_steps(steps, add_new_doc=True)
            return reply
        return self._plan_and_run(text)

    def _plan_and_run(self, text: str) -> str | None:
        ok, why = self._planner_ok()
        if not ok:
            log.info("jarvis", "not_understood", text=text, llm=why)
            return "I can only do simple commands until Ollama is running."
        context = f"Front app: {actions.frontmost_app() or 'unknown'}"
        results, reply, plan = None, None, None
        start = time.perf_counter()
        for round_ in range(3):  # plan -> act -> (look at results) -> plan the rest
            self._set("thinking", "planning\u2026")
            try:
                plan = self.planner.plan(text, context, results)
            except Exception as e:
                log.error("llm", "plan_failed", error=repr(e))
                return "My local model didn't answer."
            ok, step_reply, outputs = self._run_steps(plan.steps)
            if not ok:
                reply = step_reply
                break
            reply = plan.say or step_reply
            # When a step reports something the plan didn't expect, say that instead. Log 20260927-214838:
            # the plan said "Playing 'Buy As' by The Script" but the song wasn't in the library.
            if step_reply and any(w in step_reply.lower() for w in ("isn't", "couldn't", "not found", "no files")):
                reply = step_reply
            if plan.done or not plan.steps or not outputs:
                break
            results = outputs
        log.info("jarvis", "planned_turn", rounds=round_ + 1, ms=_ms(time.perf_counter() - start))
        self.planner.remember(text, reply or "")
        return reply

    # --- doing ---

    def _run_steps(self, steps: list[Step], add_new_doc: bool = False) -> tuple[bool, str | None, list[str]]:
        """Runs steps in order. Returns (ok, what to say, outputs for the planner)."""
        spoken, outputs, prev = [], [], None
        for step in steps:
            if step.action == "dictation":
                # "open Word and draft what I say": Word needs a blank document first
                if add_new_doc and prev and prev.action == "open_app" and \
                        prev.args.get("name", "").lower() in DOC_APPS | {"word", "ms word"}:
                    actions.run("keys", combo="cmd+n")
                    time.sleep(0.5)
                self._dictate_next = True
                prev = step
                continue
            if step.action in ("applescript", "shell"):
                code = step.args.get("script") or step.args.get("command") or ""
                risky = script_needs_confirm(code) if step.action == "applescript" else shell_needs_confirm(code)
                if risky and not self._confirm(step.args.get("summary") or f"run this {step.action}"):
                    return False, "Okay, I won't.", outputs
            self._set("thinking", f"\u2192 {step.action}")
            ok, msg = actions.run(step.action, **step.args)
            if step.action in ("find_files", "shell", "applescript"):
                outputs.append(f"{step.action} {step.args}: {msg[:1500]}")
            if not ok:
                chime("Basso", 0.4)
                return False, msg, outputs
            if msg and step.action not in ("keys", "type", "find_files", "shell", "applescript"):
                spoken.append(msg)
            prev = step
        return True, (spoken[-1] if spoken else None), outputs

    def _confirm(self, summary: str) -> bool:
        s = summary.strip().rstrip(".?")
        question = f"Should I {s[0].lower() + s[1:]}?" if s else "Should I do that?"
        log.info("jarvis", "confirm_ask", summary=summary)
        self._speak(question)
        chime("Pop", 0.3)
        self._set("listening", question + "  (yes / no)")
        answer, _ = self._listen_and_transcribe(6.0)
        yes = bool(answer) and is_yes(answer)
        log.info("jarvis", "confirm_answer", heard=answer or "", yes=yes)
        return yes

    # --- dictation ---

    def _dictate(self):
        """Type everything you say. "Scratch that", "new line", "new paragraph", "replace X with Y",
        "stop dictation". 25 s of silence also stops it."""
        self.dictating = True
        typed: list[str] = []
        log.info("jarvis", "dictation_on", app=actions.frontmost_app())
        chime("Pop", 0.6)
        while self._running:
            self._set("listening")
            audio = record_utterance(self.mic, start_timeout=25.0, end_silence=1.2, max_s=45.0)
            if audio is None:
                break
            text = self.stt(audio).strip()
            low = text.lower().strip(" .!?,")
            if not text or low in WHISPER_JUNK:
                continue
            if low in STOP_DICTATION or low.startswith(("jarvis stop", "hey jarvis stop", "stop dictation")):
                break
            if low in SCRATCH:
                if typed:
                    last = typed.pop()
                    for _ in last:
                        actions.keys("delete")
                continue
            if low in ("new line", "next line"):
                actions.keys("return")
                typed.append("\n")
                continue
            if low in ("new paragraph", "next paragraph"):
                actions.keys("return")
                actions.keys("return")
                typed.append("\n\n")
                continue
            if low.startswith(CORRECTION_STARTS) and typed and self._planner_ok()[0]:
                self._correct(typed, text)
                continue
            chunk = join_dictation("".join(typed), text)
            if chunk:
                actions.type_text(chunk)
                typed.append(chunk)
                self.reply = "".join(typed)[-80:]
        self.dictating = False
        log.info("jarvis", "dictation_off", chunks=len(typed), chars=len("".join(typed)))
        chime("Bottle", 0.4)

    def _correct(self, typed: list[str], instruction: str):
        """Spoken correction -> local LLM rewrites the last part -> only what changed is retyped."""
        self._set("thinking", "fixing\u2026")
        everything = "".join(typed)
        old = everything[-800:]
        new = self.planner.rewrite(old, instruction)
        if not new or len(new) < len(old) / 2:  # a wildly different answer would wreck the document
            log.warn("jarvis", "correction_rejected", instruction=instruction)
            return
        common = 0
        for a, b in zip(old, new):
            if a != b:
                break
            common += 1
        for _ in range(len(old) - common):
            actions.keys("delete")
        if new[common:]:
            actions.type_text(new[common:])
        typed[:] = [everything[: len(everything) - len(old)] + new]
        log.info("jarvis", "dictation_corrected", deleted=len(old) - common, typed=len(new) - common)
