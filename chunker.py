"""Streaming, "elegant" text chunking for the audio_cpp TTS tap.

`TextChunker` is fed the CLEAN CUMULATIVE reply after each yield. It detects
when enough NEW text has arrived to close a finished chunk, and returns the
newly-finished chunks (0 or more). Chunks close at natural boundaries and
never in the middle of a word, so no word is ever re-spoken or clipped.

Modes (constructor `mode=`):
- "sentence" (default, historical behaviour): the FIRST sentence boundary
  with coalesced length >= min_chars; hard-split only on a single sentence
  longer than max_chars. Byte-identical to the original implementation.
- "paragraph-greedy" ("fill toward max"): buffer closes until the pending
  length exceeds max_chars, then emit the whole buffer as one chunk — whose
  close point is, by construction, the LAST paragraph (else sentence)
  boundary under max. Inherent cost: first audio waits until the buffer
  exceeds max.
- "paragraph-lazy" ("smallest acceptable chunk"): close at the FIRST
  paragraph boundary >= min_chars and <= max_chars, falling back to the
  LAST sentence boundary <= max.

Both paragraph modes are TAG-AWARE (mirroring the audio.cpp server's
TagAware chunker, split_tag_aware_units in chunking.cpp): a `[`/`<` ...
`]`/`>` tag is an atomic unit — a boundary never lands inside an open tag,
`\n\n` inside an open tag is not a paragraph break, and tag text counts
toward the min/max budgets. (Deliberate divergence from the server: a
leading parenthetical control is NOT replicated onto every chunk.)

A single SENTENCE longer than max_chars is never hard-split while it is
still under `SOFT_SLACK_FACTOR * max_chars` (hard-coded, no config knob):
a 450-char sentence with max=400 completes whole; a 700-char one is
hard-cut.

The preprocessor keeps the prefix-consistency property (clean(a+b) starts
with clean(a)), so a chunk emitted for a prefix stays a prefix of the full
clean text and never needs re-synthesizing.

All logic here is deterministic and pure — no threads, no I/O — so it can be
unit-tested in isolation.

`feed()` returns NEW chunks; the worker synthesizes them in order.
"""

import re

# Sentence terminators. A chunk is closed when one of these is seen, provided
# the run of whitespace/punctuation after it doesn't look like an abbreviation
# (handled conservatively: we only require the terminator char).
_TERMINATORS = ".!?…:;"
# We also break on paragraph breaks for readability.
_PARAGRAPH = "\n\n"

# A single sentence longer than max is tolerated whole up to this multiple of
# max before _split_hard cuts it (soft-max slack; no config knob).
SOFT_SLACK_FACTOR = 1.5

# Default tag-open / tag-close pairs, tracked independently (as the server
# does). `()` is included because expression models (e.g. Breeze) mark
# expression with PARENTHETICALS like `(chuckle)`. User-overridable: pass
# `tag_pairs=` to TextChunker; an empty config falls back to these defaults.
DEFAULT_TAG_PAIRS = (('[', ']'), ('<', '>'), ('(', ')'))


def parse_tag_pairs(spec):
    """Parse a user config string like "() [] <>" (whitespace- or
    comma-separated two-char tokens) into a tuple of (open, close) pairs.

    Returns the DEFAULT_TAG_PAIRS for an empty/None spec, and silently skips
    malformed tokens (not exactly 2 chars)."""
    if not spec:
        return DEFAULT_TAG_PAIRS
    pairs = []
    for tok in re.split(r"[\s,]+", str(spec).strip()):
        if len(tok) == 2:
            pairs.append((tok[0], tok[1]))
    return tuple(pairs) if pairs else DEFAULT_TAG_PAIRS

_MODES = ("sentence", "paragraph-greedy", "paragraph-lazy")


def _is_number_end(text, idx):
    """True if the char at `idx` is part of an abbreviation like '3.14' or
    'e.g.' — i.e. a digit precedes a '.' that is followed by a digit."""
    if text[idx] != ".":
        return False
    if idx > 0 and text[idx - 1].isdigit() and idx + 1 < len(text) and text[idx + 1].isdigit():
        return True
    return False


def find_sentence_end(text, start, pairs=DEFAULT_TAG_PAIRS):
    """Return the index just PAST the first sentence ending at/after `start`
    within `text`, or -1 if none. The returned index is a safe split point
    (text[:ret] is a whole sentence)."""
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _TERMINATORS and not _is_number_end(text, i):
            # Consume any following CLOSING quotes/brackets only (a run of
            # extra terminators like '...' is kept: it is meaningful text).
            j = i + 1
            while j < n and text[j] in "\"')»”’":
                j += 1
            # Must be followed by whitespace/paragraph/EOF to count as a real
            # sentence end (avoid 'e.g' mid-sentence false positives are rare
            # and acceptable; the real guard is the following-space check).
            if j == n or text[j] in " \t\r\n":
                return j
            # Not followed by space: not a boundary, keep scanning from j.
            i = j
            continue
        if ch == "\n" and i + 1 < n and text[i + 1] == "\n":
            return i + 2
        i += 1
    return -1


def find_paragraph_end(text, start, pairs=DEFAULT_TAG_PAIRS):
    """Index just past the first '\\n\\n' paragraph break at/after `start`,
    or -1. Tag-aware: a '\\n\\n' INSIDE an open [..] or <..> tag is NOT a
    paragraph boundary — a multi-line tag is not a paragraph break."""
    i = text.find(_PARAGRAPH, start)
    while i != -1:
        if _tags_closed(text[:i], pairs):
            return i + 2
        i = text.find(_PARAGRAPH, i + 2)
    return -1


def _tag_counts(prefix, pairs=DEFAULT_TAG_PAIRS):
    """Unmatched open-minus-close counts per bracket type in `prefix`.
    Each bracket type is tracked independently (as the server does), so
    unbalanced nesting of one type never masks the other."""
    return [prefix.count(o) - prefix.count(c) for o, c in pairs]


def _tags_closed(prefix, pairs=DEFAULT_TAG_PAIRS):
    """True when no [..] or <..> tag is open at the end of `prefix`."""
    return all(n <= 0 for n in _tag_counts(prefix, pairs))


def _split_hard(sentence, max_chars):
    """Split a too-long sentence at whitespace (never mid-word) into pieces
    <= max_chars."""
    if len(sentence) <= max_chars:
        return [sentence]
    parts = []
    words = sentence.split(" ")
    cur = ""
    for w in words:
        # Guard against a single word longer than max_chars.
        if len(w) > max_chars:
            if cur:
                parts.append(cur)
                cur = ""
            while len(w) > max_chars:
                parts.append(w[:max_chars])
                w = w[max_chars:]
            cur = w
            continue
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            parts.append(cur)
            cur = w
    if cur:
        parts.append(cur)
    return [p for p in parts if p.strip()]


class TextChunker:
    def __init__(self, min_chars=120, max_chars=400, flush_on_end=True,
                 mode="sentence", tag_pairs=None):
        if mode not in _MODES:
            raise ValueError("unknown chunk mode %r (expected one of %r)"
                             % (mode, list(_MODES)))
        self.min_chars = max(1, min_chars)
        self.max_chars = max(self.min_chars, max_chars)
        self.mode = mode
        self._tag_pairs = parse_tag_pairs(tag_pairs)
        self._buf = ""          # raw tail: not yet recognized as a full sentence
        self._pending = []      # COMPLETE sentences coalescing toward min_chars
        self._clean_len = 0     # length of the cumulative clean text consumed
        self._finished = False  # True once flush() was called
        self._chunks = []       # finished chunks pending return to the caller

    def reset(self):
        self._buf = ""
        self._pending = []
        self._clean_len = 0
        self._finished = False
        self._chunks = []

    def _drain_new(self, new_text):
        """Consume newly appended clean text; return list of finished chunks.

        Invariant: every returned chunk contains only COMPLETE sentences, so a
        chunk boundary can never land mid-word."""
        self._buf += new_text
        chunks = []
        if self.mode == "sentence":
            chunks.extend(self._drain_sentence())
        elif self.mode == "paragraph-greedy":
            chunks.extend(self._drain_paragraph(greedy=True))
        else:  # paragraph-lazy
            chunks.extend(self._drain_paragraph(greedy=False))
        return chunks

    # -- mode: sentence (original implementation, unchanged) ------------------
    def _drain_sentence(self):
        while True:
            end = find_sentence_end(self._buf, 0, self._tag_pairs)
            if end == -1:
                break
            sentence = self._buf[:end].strip()
            self._buf = self._buf[end:]
            if not sentence:
                continue
            # A very long single sentence: flush the coalescer, then emit.
            if len(sentence) >= self.max_chars:
                if self._pending:
                    chunks = " ".join(self._pending)
                    self._pending = []
                    self._emit(chunks)
                self._extend_chunks(_split_hard(sentence, self.max_chars))
                continue
            self._pending.append(sentence)
            pending_len = sum(len(s) for s in self._pending) + max(0, len(self._pending) - 1)
            if pending_len >= self.min_chars:
                self._emit(" ".join(self._pending))
                self._pending = []
        return self._take_chunks()

    # -- modes: paragraph-greedy / paragraph-lazy (tag-aware) -----------------
    def _drain_paragraph(self, greedy):
        """Scan the buffer for a close point that is NOT inside an open tag.

        Lazy: emit each close as soon as the coalescer reaches min_chars —
        smallest acceptable chunks, lowest latency.

        Greedy: buffer closes until the pending length exceeds max_chars,
        then emit the whole buffer as one chunk. The buffer's close point is,
        by construction, the LAST paragraph (then sentence) boundary under
        max — "fill toward max". Inherent cost: first audio waits until the
        buffer exceeds max.
        """
        while True:
            end = self._paragraph_close_pos(greedy)
            if end is None:
                break
            text = self._buf[:end].strip()
            self._buf = self._buf[end:]
            if not text:
                continue
            # A close that lands over max (oversized sentence past slack, or a
            # pathological whitespace close) is word-safe-split.
            if len(text) > self.max_chars:
                if self._pending:
                    self._emit(" ".join(self._pending))
                    self._pending = []
                self._extend_chunks(_split_hard(text, self.max_chars))
                continue
            self._pending.append(text)
            if greedy:
                # Fill toward max: emit only once the buffer exceeds max.
                # (A pathological oversized close may push a chunk somewhat
                # over max; it is emitted as-is — the close rules keep this
                # bounded.)
                pending_len = sum(len(s) for s in self._pending) + max(0, len(self._pending) - 1)
                if pending_len > self.max_chars:
                    self._emit(" ".join(self._pending))
                    self._pending = []
            else:
                pending_len = sum(len(s) for s in self._pending) + max(0, len(self._pending) - 1)
                if pending_len >= self.min_chars:
                    self._emit(" ".join(self._pending))
                    self._pending = []
        return self._take_chunks()

    def _oversized_sentence_end(self):
        """If the buffer starts with a SINGLE sentence longer than max_chars
        but within max_chars * SOFT_SLACK_FACTOR, and no paragraph boundary
        precedes it, return the index just past it (close there and emit the
        sentence whole), else None."""
        end = find_sentence_end(self._buf, 0, self._tag_pairs)
        if end == -1 or not _tags_closed(self._buf[:end], self._tag_pairs):
            return None
        if find_paragraph_end(self._buf, 0, self._tag_pairs) != -1:
            return None
        length = len(self._buf[:end].strip())
        if self.max_chars < length <= self.max_chars * SOFT_SLACK_FACTOR:
            return end
        return None

    def _paragraph_close_pos(self, greedy):
        """Index just past which the current chunk should close, or None to
        keep accumulating. Tag-aware: candidates inside an open tag are
        skipped (held until the tag closes)."""
        buf = self._buf
        if greedy:
            # Fill toward max: last paragraph boundary at/under max.
            end = self._last_valid_boundary(find_paragraph_end, self.max_chars)
            if end is not None:
                return end
            # A single oversized sentence within the soft-max slack: emit it
            # whole rather than cutting it mid-word at max.
            end = self._oversized_sentence_end()
            if end is not None:
                return end
            # No paragraph boundary under max: last sentence boundary under
            # max.
            end = self._last_valid_boundary(find_sentence_end, self.max_chars)
            if end is not None:
                return end
            # Nothing usable: once past the soft-max slack, close at the last
            # whitespace before max (never mid-word); the tail stays buffered.
            if len(buf) > self.max_chars * SOFT_SLACK_FACTOR:
                return self._whitespace_close_pos()
            return None
        # lazy: first paragraph boundary >= min and <= max.
        end = find_paragraph_end(buf, 0, self._tag_pairs)
        while end != -1 and (end < self.min_chars
                              or not _tags_closed(buf[:end], self._tag_pairs)):
            end = find_paragraph_end(buf, end, self._tag_pairs)
        if end != -1 and end <= self.max_chars:
            return end
        # A single oversized sentence within the soft-max slack: emit whole.
        end = self._oversized_sentence_end()
        if end is not None:
            return end
        # No acceptable paragraph boundary so far. The sentence fallback may
        # only fire once the buffer has passed max_chars: below max, a
        # paragraph boundary may simply not have ARRIVED YET (streaming), so
        # closing at a sentence here would preempt the next \n\n and produce
        # ragged sub-paragraph chunks.
        if len(buf) > self.max_chars:
            # Confirmed: no paragraph boundary in [min, max]. LAST sentence
            # boundary in [min, max] (lazy: never close below min — sub-min
            # sentences coalesce). No such boundary: keep accumulating (the
            # flush emits the tail at end of reply, word-safe).
            end = self._last_valid_boundary_in(find_sentence_end, self.min_chars, self.max_chars)
            if end is not None:
                return end
        # A single over-long sentence stays whole up to the soft-max slack,
        # then closes at the last whitespace before max (never mid-word); the
        # tail stays buffered.
        if len(buf) > self.max_chars * SOFT_SLACK_FACTOR:
            return self._whitespace_close_pos()
        return None

    def _whitespace_close_pos(self):
        """Close position for an over-slack buffer: the last space at/under
        max_chars, so the close is never mid-word. Falls back to a bare char
        cut at max (for a wall of unbroken text) which _split_hard later
        re-splits word-safely."""
        limit = self.max_chars
        space = self._buf.rfind(" ", 0, limit)
        if space > limit // 2:  # don't leave a degenerate micro-first-chunk
            return space + 1
        return limit

    def _last_valid_boundary(self, finder, limit):
        """Last boundary (via `finder`) at position <= `limit` that is not
        inside an open tag; None if none."""
        return self._last_valid_boundary_in(finder, 0, limit)

    def _last_valid_boundary_in(self, finder, lo, limit):
        """Last tag-closed boundary in [lo, limit]; None if none."""
        end = finder(self._buf, 0, self._tag_pairs)
        last = None
        while end != -1 and end <= limit:
            if end >= lo and _tags_closed(self._buf[:end], self._tag_pairs):
                last = end
            end = finder(self._buf, end, self._tag_pairs)
        return last

    # -- emission helpers -----------------------------------------------------
    def _emit(self, chunk):
        self._chunks.append(chunk)

    def _extend_chunks(self, parts):
        self._chunks.extend(parts)

    def _take_chunks(self):
        c, self._chunks = self._chunks, []
        return c

    def _new_drain(self):
        self._chunks = []
        return self._drain_new

    # -- core feed ------------------------------------------------------------
    def feed(self, cumulative_clean):
        """Feed the cumulative CLEAN reply. Returns newly finished chunks."""
        if self._finished:
            return []
        # cumulative_clean must grow from what we last saw; if a shorter/diff
        # prefix arrives (rare: regeneration), resync by resetting.
        if len(cumulative_clean) < self._clean_len:
            self.reset()
            self._clean_len = 0
        new_text = cumulative_clean[self._clean_len:]
        self._clean_len = len(cumulative_clean)
        if not new_text:
            return []
        return self._drain_new(new_text)

    def flush(self):
        """End of reply: emit any remaining buffered text as a final chunk."""
        if self._finished:
            return []
        self._finished = True
        tail = self._buf.strip()
        # A tag still open at end of stream is closed with its matching
        # bracket (the server's tag_close_suffix) so the final chunk is never
        # left with an unbalanced tag.
        for o, c in reversed(self._tag_pairs):
            n = tail.count(o) - tail.count(c)
            if n > 0:
                tail += c * n
        leftover = " ".join(self._pending + ([tail] if tail else [])).strip()
        self._pending = []
        self._buf = ""
        if not leftover:
            return []
        if len(leftover) >= self.max_chars:
            return _split_hard(leftover, self.max_chars)
        return [leftover]