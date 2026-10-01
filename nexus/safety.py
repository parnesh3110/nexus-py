"""Which scripts and shell commands need a spoken "yes" first. Pure, unit-tested."""

RISKY_SCRIPT_WORDS = ("delete", "do shell script", "empty trash", "empty the trash", "erase", "restart",
                      "shut down", "log out", "trash", "send", "sudo", "format", "remove")

SAFE_SHELL = {"ls", "pwd", "whoami", "date", "cal", "uptime", "df", "du", "sw_vers", "system_profiler", "mdfind",
              "mdls", "cat", "head", "tail", "wc", "which", "ps", "top", "ipconfig", "echo", "find", "grep", "file",
              "stat", "sysctl", "ping", "uname", "hostname", "id", "sort", "uniq", "awk", "cut", "open", "pmset"}
SAFE_SHELL_PREFIXES = ("defaults read", "pmset -g", "git status", "git log", "git diff", "brew list", "brew info",
                       "networksetup -get", "shortcuts list")
RISKY_SHELL_BITS = (";", "&&", "||", ">", "`", "$(", "-delete", "-exec", "rm ", "sudo", "mv ", "chmod", "kill")


def script_needs_confirm(script: str) -> bool:
    s = script.lower()
    return any(w in s for w in RISKY_SCRIPT_WORDS)


def shell_needs_confirm(cmd: str) -> bool:
    c = cmd.strip()
    if any(b in c for b in RISKY_SHELL_BITS):
        return True
    for segment in c.split("|"):  # every piped part must start with a read-only command
        s = segment.strip()
        if s.startswith(SAFE_SHELL_PREFIXES):
            continue
        first = s.split(" ")[0] if s else ""
        if first not in SAFE_SHELL:
            return True
    return False


YES = ("yes", "yeah", "yep", "yup", "sure", "do it", "go ahead", "ok", "okay", "confirm", "please do", "correct",
       "go for it", "affirmative")


def is_yes(text: str) -> bool:
    t = text.lower().strip(" .!?,")
    if t.startswith("no") or "don't" in t or "do not" in t or "cancel" in t:
        return False
    return t.startswith(YES)
