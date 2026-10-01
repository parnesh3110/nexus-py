"""Command parsing, including sentences that failed in earlier logs."""
from nexus.commands import Step, parse, split_steps, strip_fillers


def test_fillers_and_misheard_wake_word():
    assert strip_fillers("Can you can you open Apple Music?") == "open Apple Music"
    assert strip_fillers("Cherish, can you play the reason on Apple Music?") == "play the reason on Apple Music"
    assert strip_fillers("okay please minimize") == "minimize"


def test_exact_commands():
    assert parse("minimize") == [Step("window", {"action": "minimize"})]
    assert parse("Skip this song.") == [Step("music", {"action": "next"})]
    assert parse("new tab") == [Step("keys", {"combo": "cmd+t"})]


def test_commands_with_arguments():
    assert parse("open Apple Music") == [Step("open_app", {"name": "Apple Music"})]
    assert parse("Cherish, can you play the reason on Apple Music?") == [
        Step("music", {"action": "play", "query": "the reason"})]
    assert parse("type Hello World") == [Step("type", {"text": "Hello World"})]
    assert parse("go to youtube") == [Step("open_url", {"url": "youtube.com"})]


def test_multi_step():
    assert parse("open Apple Music and play the reason") == [
        Step("open_app", {"name": "Apple Music"}), Step("music", {"action": "play", "query": "the reason"})]
    assert parse("Open Safari, then go to github") == [
        Step("open_app", {"name": "Safari"}), Step("open_url", {"url": "github.com"})]


def test_and_inside_an_argument_is_not_split():
    assert split_steps("search for salt and pepper") == ["search for salt and pepper"]
    assert parse("search for salt and pepper") == [Step("open_url", {"url": "salt and pepper"})]


def test_unknown_goes_to_the_smarter_layer():
    assert parse("what's the weather like") is None
    assert parse("open Word and write me a cover letter") is None  # second step isn't a simple command
