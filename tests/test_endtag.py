"""End-tag detection + streaming suppression.

Workflow: copy this file to the textgen root and run `python3 test_endtag.py`
there (so `modules` and `extensions.audio_cpp` import as in the server).
"""
from extensions.audio_cpp import preprocessor
from modules.reasoning import THINKING_FORMATS

opener = "<" + "think" + ">"
closer = "</" + "think" + ">"
g_opener = "<|" + "channel" + ">thought"
g_closer = "<" + "channel" + "|>"

# --- raw-template detection (fallback) ---
assert preprocessor.detect_end_tag("Hello {bot}\n" + opener + " {{resp}}") == closer, "think-embed"
assert preprocessor.detect_end_tag("A " + g_opener + " B") == g_closer, "gemma-embed"
assert preprocessor.detect_end_tag("system: answer\n" + closer) == closer, "bare-closer"
assert preprocessor.detect_end_tag("User: {user}\nAssistant: {bot}") == "", "plain"
assert preprocessor.detect_end_tag("") == "", "empty"

# --- rendered-prompt detection (primary) ---
r = preprocessor.detect_end_tag_rendered
# enable_thinking resolved TRUE: last opener unclosed -> thinking mode
assert r("sys\n" + opener + "\n") == closer, "rendered: thinking on"
# enable_thinking resolved FALSE: pair rendered -> closed -> NOT thinking
assert r("sys\n" + opener + "\n\n" + closer + "\n\n") == "", "rendered: thinking off"
# latest opener wins over earlier closed pairs
assert r(opener + closer + "\n" + opener + "\n") == closer, "rendered: latest wins"
# no opener, bare closer -> end-only
assert r("sys\n" + closer + "\n") == closer, "rendered: bare closer"
# nothing
assert r("sys\nUser: hi\nAssistant: ") == "", "rendered: plain"
assert r("") == "", "rendered: empty"

# --- streaming suppression ---
t1 = "Thinking Process: 1. Analyze the request"
assert preprocessor.clean(t1, thinking_end_tag=closer) == "", "suppress pre-tag"
t2 = t1 + "\n" + closer + " YES, I ACKNOWLEDGE."
assert preprocessor.clean(t2, thinking_end_tag=closer) == "YES, I ACKNOWLEDGE.", "split at tag"
assert preprocessor.clean(t1, thinking_end_tag="") != "", "no-tag passthrough"

print("ALL DETECTION+SUPPRESSION TESTS PASSED (%d formats loaded)" % len(THINKING_FORMATS))
