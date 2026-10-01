"""Local LLM planner: request -> JSON plan of actions, via Ollama on this Mac (no cloud).

The model gets the list of actions it can use and a few examples, and must answer in a
JSON schema (Ollama structured output), so the reply always parses. If it needs information
first (file search, a shell command), it says done=false, gets the results, and plans the rest.

    brew install ollama && ollama serve   # once
    ollama pull qwen2.5:3b                # ~2 GB
"""
import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field

from .commands import Step
from .log import log

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("NEXUS_LLM", "qwen2.5:3b")

# action -> (allowed args, required args)
ACTIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "open_app": (("name",), ("name",)),
    "quit_app": (("name",), ("name",)),
    "focus_app": (("name",), ("name",)),
    "keys": (("combo",), ("combo",)),
    "type": (("text",), ("text",)),
    "window": (("action",), ("action",)),
    "music": (("action", "query"), ("action",)),
    "open_url": (("url",), ("url",)),
    "system": (("action",), ("action",)),
    "menu": (("path", "app"), ("path",)),
    "click": (("name",), ("name",)),
    "applescript": (("script", "summary"), ("script",)),
    "shell": (("command", "summary"), ("command",)),
    "find_files": (("query",), ("query",)),
    "open": (("target", "app"), ("target",)),
    "dictation": ((), ()),
}

SCHEMA = {
    "type": "object",
    "properties": {
        "say": {"type": "string"},
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": list(ACTIONS)}, "args": {"type": "object"}},
            "required": ["action", "args"]}},
        "done": {"type": "boolean"},
    },
    "required": ["say", "steps", "done"],
}

SYSTEM = """You are Jarvis, a voice assistant that controls the user's Mac. Turn each request into steps.
Answer ONLY with JSON: {"say": "short spoken reply", "steps": [{"action": "...", "args": {...}}], "done": true}

Actions:
- open_app {name}: open or switch to an app. It is in front afterwards.
- quit_app {name}
- keys {combo}: keyboard shortcut, e.g. "cmd+t", "cmd+shift+n", "return"
- type {text}: type text into the focused app
- window {action}: minimize | maximize | fullscreen | close | left_half | right_half
- music {action, query}: play (with query) | pause | resume | next | previous
- open_url {url}: a website, or search words
- system {action}: volume_up | volume_down | mute | unmute | lock_screen
- menu {path, app}: click a menu bar item, path like ["File", "New Window"]
- click {name}: click a button by its label in the front window
- applescript {script, summary}: AppleScript for anything an app can do (Notes, Reminders, Calendar, Mail, Finder, Safari, System Events)
- shell {command, summary}: a zsh command, for system info or files
- find_files {query}: find files by name. You get the paths back.
- open {target, app}: open a file path or URL, optionally in a specific app
- dictation {}: type everything the user says next. Use when they want to write by speaking.

Rules:
- Do every part of the request, in order.
- When the user asks you to write something (an email, a list, a note), write the text yourself and use type.
- A question gets an answer in "say" and no steps.
- "say" is spoken aloud: under 25 words, no markdown.
- If you need a result before you can finish (find_files, shell), set "done": false. You will get the results and plan the rest.

Examples:
User: open apple music and play the reason
{"say": "Playing The Reason.", "steps": [{"action": "open_app", "args": {"name": "Music"}}, {"action": "music", "args": {"action": "play", "query": "The Reason"}}], "done": true}
User: open word and draft what I say
{"say": "", "steps": [{"action": "open_app", "args": {"name": "Microsoft Word"}}, {"action": "keys", "args": {"combo": "cmd+n"}}, {"action": "dictation", "args": {}}], "done": true}
User: open notes and write a shopping list with milk eggs and bread
{"say": "Done.", "steps": [{"action": "open_app", "args": {"name": "Notes"}}, {"action": "keys", "args": {"combo": "cmd+n"}}, {"action": "type", "args": {"text": "Shopping list\\n- Milk\\n- Eggs\\n- Bread"}}], "done": true}
User: turn on dark mode
{"say": "Dark mode on.", "steps": [{"action": "applescript", "args": {"script": "tell application \\"System Events\\" to tell appearance preferences to set dark mode to true", "summary": "turn on dark mode"}}], "done": true}
User: open my resume
{"say": "", "steps": [{"action": "find_files", "args": {"query": "resume"}}], "done": false}
User: what's the capital of japan
{"say": "Tokyo.", "steps": [], "done": true}"""


@dataclass
class Plan:
    say: str = ""
    steps: list[Step] = field(default_factory=list)
    done: bool = True
    dropped: list[str] = field(default_factory=list)  # steps the model got wrong


def validate(obj: dict) -> Plan:
    """Keep only well-formed steps with known actions and their allowed args (so a model that
    invents an argument can't crash an action)."""
    plan = Plan(say=str(obj.get("say") or "").strip(), done=bool(obj.get("done", True)))
    for raw in obj.get("steps") or []:
        if not isinstance(raw, dict):
            continue
        action, args = raw.get("action"), raw.get("args") or {}
        if action not in ACTIONS or not isinstance(args, dict):
            plan.dropped.append(str(action))
            continue
        allowed, required = ACTIONS[action]
        clean = {k: v for k, v in args.items() if k in allowed and v not in (None, "")}
        if action == "menu" and isinstance(clean.get("path"), str):
            clean["path"] = [p.strip() for p in clean["path"].split(">")]
        if any(r not in clean for r in required):
            plan.dropped.append(action)
            continue
        plan.steps.append(Step(action, clean))
    plan.steps = plan.steps[:10]
    return plan


class Planner:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.history: deque = deque(maxlen=6)  # recent (user, jarvis) messages, for follow-ups

    def _post(self, path: str, body: dict, timeout: float) -> dict:
        req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def available(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(HOST + "/api/tags", timeout=1.5) as r:
                names = [m["name"] for m in json.loads(r.read()).get("models", [])]
        except Exception:
            return False, "Ollama isn't running. Start it with: ollama serve"
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            return False, f"Model missing. Run: ollama pull {self.model}"
        return True, "ok"

    def warm_up(self):
        """Load the model into memory now, so the first real request isn't slow."""
        try:
            start = time.perf_counter()
            self._post("/api/generate", {"model": self.model, "prompt": "hi", "stream": False,
                                         "keep_alive": "30m", "options": {"num_predict": 1}}, 60)
            log.info("llm", "warm", model=self.model, ms=int((time.perf_counter() - start) * 1000))
        except Exception as e:
            log.warn("llm", "warm_failed", error=str(e))

    def plan(self, text: str, context: str, results: list[str] | None = None) -> Plan:
        messages = [{"role": "system", "content": SYSTEM}]
        for role, content in self.history:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": f"[{context}]\n{text}"})
        if results:
            messages.append({"role": "user", "content": "Results of your steps:\n" + "\n".join(results)
                             + "\nNow plan the rest (done: true when finished)."})
        start = time.perf_counter()
        resp = self._post("/api/chat", {"model": self.model, "messages": messages, "stream": False,
                                        "format": SCHEMA, "keep_alive": "30m",
                                        "options": {"temperature": 0.1, "num_predict": 400}}, 60)
        raw = resp.get("message", {}).get("content", "")
        try:
            plan = validate(json.loads(raw))
        except json.JSONDecodeError:
            plan = Plan(say=raw.strip()[:200])
        log.info("llm", "plan", model=self.model, ms=int((time.perf_counter() - start) * 1000),
                 steps=[s.action for s in plan.steps], dropped=plan.dropped, done=plan.done,
                 say=plan.say[:120], tokens=resp.get("eval_count"))
        return plan

    def remember(self, user: str, reply: str):
        self.history.append(("user", user))
        self.history.append(("assistant", reply or "(done)"))

    def rewrite(self, text: str, instruction: str) -> str | None:
        """Apply a spoken correction to dictated text ("replace Monday with Tuesday")."""
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        prompt = ("Apply the correction to the text. Change nothing else; keep punctuation and line breaks.\n"
                  f"Text:\n{text}\n\nCorrection: {instruction}")
        try:
            resp = self._post("/api/chat", {"model": self.model, "stream": False, "format": schema,
                                            "keep_alive": "30m", "options": {"temperature": 0},
                                            "messages": [{"role": "user", "content": prompt}]}, 30)
            return json.loads(resp["message"]["content"])["text"]
        except Exception as e:
            log.warn("llm", "rewrite_failed", error=str(e))
            return None
