"""Text cleanup before TTS.

Designed for the streaming tap: `clean` is applied to the CUMULATIVE bot
reply and must be idempotent and prefix-consistent (the clean form of the
growing text is always a growing text; the chunker detects the divergence
point). No cross-delta state of its own.
"""

import re

_FALLBACK_THINKING = re.compile(r"<\s*/?\s*think(?:ing)?\s*/?\s*>.*?"
                                 r"(?:</\s*think(?:ing)?\s*>|$)",
                                 re.DOTALL | re.IGNORECASE)


def _extract_reasoning(text):
    # Textgen's own Thought-accordion extraction (modules/reasoning.py) —
    # the single source of truth for thinking-block formats: </think>,
    # GPT-OSS <|channel|>analysis, Gemma 4 <|channel>thought…<channel|>,
    # Solar, Qwen3-next end-only, streaming partial-tag suppression, etc.
    # Imported lazily so the module also loads outside the textgen tree
    # (standalone unit tests); a minimal  fallback covers that case.
    try:
        from modules import reasoning
    except ImportError:
        return (None, _FALLBACK_THINKING.sub("", text).strip())
    return reasoning.extract_reasoning(text)

_CITATIONS = re.compile(r"\[\[\d+(?:[,-]\d+)*\]\]")
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})([^*\n]+)\1")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_NEWLINES = re.compile(r"\n{3,}")


def detect_end_tag(template_text):
    """Which thinking CLOSE tag does this model's output use, judged from its
    instruction/chat template ('' = none detected).

    Needed because for some models the OPENING tag is embedded in the
    instruction template, so the output never contains it — only the closing
    tag.  extract_reasoning cannot detect such blocks while streaming (with
    no close tag yet the text is indistinguishable from an ordinary reply),
    so the caller must know the close tag up front and suppress everything
    until it appears.

    Derived from modules.reasoning.THINKING_FORMATS (the same table the
    Thought accordion uses): for a (start, end, content) entry, if the
    template contains `start` (opener embedded) or, for start-less
    end-only entries, `end` itself, the output's close tag is `end`.
    """
    if not template_text:
        return ""
    try:
        from modules.reasoning import THINKING_FORMATS
    except Exception:
        return ""
    for start, end, _content in THINKING_FORMATS:
        probe = start if start else end
        if probe and probe in template_text:
            return end
    return ""


def detect_end_tag_rendered(prompt_text):
    """Which thinking CLOSE tag to expect in the output, judged from a FULLY
    RENDERED prompt (Jinja conditionals already resolved with the active
    settings — e.g. ``enable_thinking``).

    The raw-template scan above cannot see inside conditionals, so a
    template whose thinking branch is inactive would still "detect" a close
    tag and every (thinking-free) reply would be suppressed. Rules:
      1. Latest opener in the prompt:
         - a close tag appears after it → opened-and-closed in the prompt
           (thinking-off templates render the pair) → no suppression;
         - no close tag after it → model is in thinking mode → that close.
      2. No opener, but a bare close tag present → end-only model → it.
      3. Neither → "" (no suppression).
    """
    if not prompt_text:
        return ""
    try:
        from modules.reasoning import THINKING_FORMATS
    except Exception:
        return ""
    last_open = -1
    last_end = ""
    for start, end, _c in THINKING_FORMATS:
        if not start:
            continue
        pos = prompt_text.rfind(start)
        if pos != -1 and pos > last_open:
            last_open = pos
            last_end = end
    if last_open != -1:
        if prompt_text.find(last_end, last_open) != -1:
            return ""
        return last_end
    for start, end, _c in THINKING_FORMATS:
        if start is None and end and end in prompt_text:
            return end
    return ""


def clean(text, strip_thinking=True, strip_markdown=True, strip_citations=True,
          thinking_end_tag=""):
    if not text:
        return ""
    out = text
    if strip_citations:
        out = _CITATIONS.sub("", out)
    if strip_thinking:
        if thinking_end_tag and thinking_end_tag not in out:
            # End-only thinking model: the closing tag hasn't appeared yet, so
            # all the visible text is (presumed) thinking.
            # Suppress until the tag appears. Once the tag is present, extract_reasoning splits correctly.
            return ""
        _thinking, out = _extract_reasoning(out)
    if strip_markdown:
        out = _IMAGE.sub(lambda m: m.group(1) or "", out)
        out = _CODE_FENCE.sub(" ", out)
        out = _INLINE_CODE.sub(lambda m: m.group(0)[1:-1], out)
        out = _MARKDOWN_LINK.sub(r"\1", out)
        out = _BOLD_ITALIC.sub(r"\2", out)
        out = _HTML_COMMENT.sub("", out)
        out = _HTML_TAG.sub("", out)
        out = out.replace("**", "").replace("__", "")
    out = _NEWLINES.sub("\n\n", out)
    return out