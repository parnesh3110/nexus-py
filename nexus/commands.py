"""What you said -> a list of actions. Pure (no macOS calls), so it's unit-tested.

Handles exact phrases ("minimize", "next song"), phrases with an argument ("open Word",
"play The Reason", "type hello world") and several steps in one sentence ("open Apple Music
and play The Reason"). Anything else goes to the local intent model / LLM in later phases.
"""
import re
from dataclasses import dataclass, field


@dataclass
class Step:
    action: str
    args: dict = field(default_factory=dict)


FILLERS = ["can you ", "could you ", "would you ", "will you ", "please ", "hey ", "okay ", "ok ", "um ",
           "uh ", "so ", "jarvis ", "hey jarvis ", "i want you to ", "i need you to ",
           "and you ", "and can you ", "and "]  # Whisper heard "can you" as "and you" (log 20260924-034234)


def strip_fillers(s: str) -> str:
    s = s.strip().strip(".!?").strip()
    # One word before a comma is someone being addressed ("Jarvis, ..." or a mishearing of it)
    if "," in s and " " not in s[: s.index(",")] and s.index(",") <= 10:
        s = s[s.index(",") + 1:].strip()
    changed = True
    while changed:
        changed = False
        low = s.lower()
        for f in FILLERS:
            if low.startswith(f):
                s = s[len(f):]
                changed = True
                break
    s = s.replace(" please", "").strip(" ,.!?")
    return s


def _k(combo): return Step("keys", {"combo": combo})
def _w(action): return Step("window", {"action": action})
def _m(action): return Step("music", {"action": action})
def _s(action): return Step("system", {"action": action})


EXACT: dict[str, Step] = {}
for phrases, step in [
    (["minimize", "minimize this", "minimize it", "minimize the window", "minimize window"], _w("minimize")),
    (["maximize", "maximize this", "maximize it", "maximize the window", "make it bigger"], _w("maximize")),
    (["full screen", "fullscreen", "make it full screen", "go full screen", "exit full screen"], _w("fullscreen")),
    (["close this", "close it", "close the window", "close window", "close this window"], _w("close")),
    (["left half", "snap left", "move it left", "move this left"], _w("left_half")),
    (["right half", "snap right", "move it right", "move this right"], _w("right_half")),
    (["new tab", "open a new tab", "open new tab"], _k("cmd+t")),
    (["close tab", "close this tab", "close the tab"], _k("cmd+w")),
    (["new window", "open a new window"], _k("cmd+n")),
    (["new note", "new document", "new file", "create a new note", "make a new note"], _k("cmd+n")),
    (["go back", "back"], _k("cmd+[")),
    (["go forward", "forward"], _k("cmd+]")),
    (["refresh", "reload", "reload the page"], _k("cmd+r")),
    (["undo", "undo that"], _k("cmd+z")),
    (["redo"], _k("cmd+shift+z")),
    (["copy", "copy that", "copy this"], _k("cmd+c")),
    (["paste", "paste it", "paste that"], _k("cmd+v")),
    (["select all"], _k("cmd+a")),
    (["save", "save it", "save this"], _k("cmd+s")),
    (["scroll down", "page down"], _k("pagedown")),
    (["scroll up", "page up"], _k("pageup")),
    (["pause", "pause music", "pause the music", "stop the music", "pause the song"], _m("pause")),
    (["resume", "resume music", "resume the music", "play music", "continue playing", "play", "play it"], _m("resume")),
    (["next song", "next track", "skip", "skip this song", "skip song"], _m("next")),
    (["previous song", "previous track", "last song"], _m("previous")),
    (["mute", "mute the sound"], _s("mute")),
    (["unmute"], _s("unmute")),
    (["volume up", "louder", "turn it up", "turn up the volume"], _s("volume_up")),
    (["volume down", "quieter", "turn it down", "softer", "turn down the volume"], _s("volume_down")),
    (["lock screen", "lock my mac", "lock the screen"], _s("lock_screen")),
    (["quit this", "quit this app", "quit the app", "close this app"], Step("quit_app", {"name": "this"})),
    (["draft what i'm saying", "draft what im saying", "draft what i say", "type what i say", "type what i'm saying",
      "write what i say", "write what i'm saying", "take dictation", "dictate", "start dictation", "start dictating",
      "take notes", "write this down", "draft", "dictation", "dictation mode"], Step("dictation")),
]:
    for p in phrases:
        EXACT[p] = step

# prefix -> (action, arg name)
PREFIXES = [
    (("open ", "launch ", "start "), "open_app", "name"),
    (("switch to ", "go to the app "), "focus_app", "name"),
    (("quit ", "close the app "), "quit_app", "name"),
    (("type ",), "type", "text"),
    (("play ", "put on ", "listen to ", "i want to listen to "), "music_play", "query"),
    (("search for ", "search ", "google ", "look up "), "open_url", "url"),
    (("go to ",), "go_to", "site"),
]

STEP_VERBS = ("open", "launch", "start", "switch", "quit", "close", "type", "write", "play", "put on", "listen",
              "search", "google", "look up", "go", "minimize", "maximize", "full screen", "new", "create", "make",
              "pause", "resume", "next", "skip", "previous", "mute", "unmute", "volume", "lock", "undo", "copy",
              "paste", "save", "select", "scroll", "refresh", "reload", "snap", "move", "draft", "dictate", "take")


def parse_one(text: str) -> Step | None:
    t = strip_fillers(text)
    low = t.lower()
    if low in EXACT:
        return EXACT[low]
    for prefixes, action, arg in PREFIXES:
        for p in prefixes:
            if low.startswith(p) and len(t) > len(p):
                value = t[len(p):].strip()
                if action == "music_play":
                    # "on YouTube Music" / "on Spotify": Jarvis plays through Apple Music, so drop the service
                    for tail in (" on apple music", " in apple music", " on youtube music", " on youtube",
                                 " on spotify", " on music", " in music", " for me"):
                        if value.lower().endswith(tail):
                            value = value[: -len(tail)]
                    return Step("music", {"action": "play", "query": value})
                if action == "go_to":
                    site = value.lower().replace(" ", "")
                    return Step("open_url", {"url": site if "." in site else site + ".com"})
                if action in ("open_app", "focus_app", "quit_app"):
                    v = value.lower()
                    # "open Notes and write a shopping list..." is not an app called "Notes and write..."
                    if " and " in v or " then " in v or len(v.split()) > 4:
                        return None
                return Step(action, {arg: value})
    return None


def split_steps(text: str) -> list[str]:
    """Split on and/then only when every piece starts with a command verb, so
    "search for salt and pepper" stays one step."""
    t = strip_fillers(text)
    pieces = [t]
    for sep in (", and then ", " and then ", ", then ", " then ", " after that ", ". ", ", and ", " and "):
        pieces = [p for piece in pieces for p in piece.split(sep)]
    pieces = [p.strip(" ,.") for p in pieces if p.strip(" ,.")]
    if len(pieces) >= 2 and all(p.lower().startswith(STEP_VERBS) for p in pieces):
        return pieces
    return [t]


def parse(text: str) -> list[Step] | None:
    """All steps, or None if any step isn't understood (then a smarter model gets it)."""
    steps = [parse_one(p) for p in split_steps(text)]
    return None if not steps or any(s is None for s in steps) else steps


SPOKEN_PUNCTUATION = {"comma": ",", "full stop": ".", "period": ".", "question mark": "?",
                      "exclamation mark": "!", "exclamation point": "!", "colon": ":", "semicolon": ";"}
# Whole words only ("command" and "periodic" stay as they are), with any punctuation Whisper put after them
_SPOKEN_RE = re.compile(r"\s*\b(" + "|".join(sorted(SPOKEN_PUNCTUATION, key=len, reverse=True)) + r")\b[.,]?",
                        re.IGNORECASE)


def join_dictation(before: str, chunk: str) -> str:
    """Spacing and capitals for a new dictated chunk, relative to what's already typed."""
    chunk = _SPOKEN_RE.sub(lambda m: SPOKEN_PUNCTUATION[m.group(1).lower()], chunk)
    chunk = re.sub(r"([,.;:!?])[,.]+", r"\1", chunk)  # "periodically, comma" -> one comma, not two
    chunk = chunk.strip()
    if not chunk:
        return ""
    last = before[-1:] if before else ""
    if not before or last in ".!?\n":
        chunk = chunk[0].upper() + chunk[1:]
    elif chunk[0].isupper() and chunk[1:2].islower() and not chunk.startswith(("I ", "I'")):
        chunk = chunk[0].lower() + chunk[1:]  # each pause starts a new transcription, which capitalizes
    if before and last not in " \n" and chunk[0] not in ",.;:!?":
        chunk = " " + chunk
    return chunk
