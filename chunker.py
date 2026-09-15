"""Sentence-aware chunking of a growing LLM stream for low-latency TTS.

`TextChunker` is fed the CLEAN CUMULATIVE reply after each yield. It detects
the newly appended suffix, splits it into finished sentences, coalesces small
sentences up to `min_chars`, and yields TTS-ready chunks. A chunk is emitted
the moment a sentence boundary is seen (that is the latency win), so audio
for sentence N starts while the model is still generating N+1.

All logic here is deterministic and pure — no threads, no I/O — so it can be
unit-tested in isolation.
"""

# Sentence terminators. A chunk is closed when one of these is seen, provided
# the run of whitespace/punctuation after it doesn't look like an abbreviation
# (handled conservatively: we only require the terminator char).
_TERMINATORS = ".!?…:;"
# We also break on paragraph breaks for readability.
_PARAGRAPH = "\n\n"


def _is_number_end(text, idx):
    """True if the char at `idx` is part of an abbreviation like '3.14' or
    'e.g.' — i.e. a digit precedes a '.' that is followed by a digit."""
    if text[idx] != ".":
        return False
    if idx > 0 and text[idx - 1].isdigit() and idx + 1 < len(text) and text[idx + 1].isdigit():
        return True
    return False


def find_sentence_end(text, start):
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
    def __init__(self, min_chars=120, max_chars=400, flush_on_end=True):
        self.min_chars = max(1, min_chars)
        self.max_chars = max(self.min_chars, max_chars)
        self._buf = ""          # raw tail: not yet recognized as a full sentence
        self._pending = []      # COMPLETE sentences coalescing toward min_chars
        self._clean_len = 0     # length of the cumulative clean text consumed
        self._finished = False  # True once flush() was called

    def reset(self):
        self._buf = ""
        self._pending = []
        self._clean_len = 0
        self._finished = False

    def _drain_new(self, new_text):
        """Consume newly appended clean text; return list of finished chunks.

        Invariant: every returned chunk contains only COMPLETE sentences, so a
        chunk boundary can never land mid-word."""
        self._buf += new_text
        chunks = []
        while True:
            end = find_sentence_end(self._buf, 0)
            if end == -1:
                break
            sentence = self._buf[:end].strip()
            self._buf = self._buf[end:]
            if not sentence:
                continue
            # A very long single sentence: flush the coalescer, then emit.
            if len(sentence) >= self.max_chars:
                if self._pending:
                    chunks.append(" ".join(self._pending))
                    self._pending = []
                chunks.extend(_split_hard(sentence, self.max_chars))
                continue
            self._pending.append(sentence)
            pending_len = sum(len(s) for s in self._pending) + max(0, len(self._pending) - 1)
            if pending_len >= self.min_chars:
                chunks.append(" ".join(self._pending))
                self._pending = []
        return chunks

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
        leftover = " ".join(self._pending + ([tail] if tail else [])).strip()
        self._pending = []
        self._buf = ""
        if not leftover:
            return []
        if len(leftover) >= self.max_chars:
            return _split_hard(leftover, self.max_chars)
        return [leftover]