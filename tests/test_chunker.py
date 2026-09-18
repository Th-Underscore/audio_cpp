import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chunker
import preprocessor as pp


def test_basic_sentences():
    c = chunker.TextChunker(min_chars=10, max_chars=40)
    out = []
    acc = ""
    for s in ["Hello world. ", "How are you? ", "Fine! "]:
        acc += s
        out += c.feed(acc)
    out += c.flush()
    assert out == ["Hello world.", "How are you?", "Fine!"], out
    print("basic ok", out)


def test_coalesce_short():
    c = chunker.TextChunker(min_chars=30, max_chars=200)
    out = []
    acc = ""
    for s in ["Hi. ", "How are you? ", "I am well. "]:
        acc += s
        out += c.feed(acc)
    out += c.flush()
    assert out, out
    joined = " ".join(out)
    assert "Hi." in joined and "How are you?" in joined and "I am well." in joined
    for ch in out:
        assert len(ch) <= 201, (ch, len(ch))
    print("coalesce ok", out)


def test_long_sentence_hard_split():
    c = chunker.TextChunker(min_chars=10, max_chars=20)
    words = " ".join(["word"] * 20)
    out = c.feed(words + ". ")
    out += c.flush()
    for ch in out:
        assert len(ch) <= 20, (ch, len(ch))
    print("hard split ok", out)


def test_number_decimal_no_split():
    c = chunker.TextChunker(min_chars=5, max_chars=100)
    out = c.feed("Pi is 3.14 right? Yes. ")
    out += c.flush()
    joined = " ".join(out)
    assert "3.14" in joined, out
    print("decimal ok", out)


def test_paragraph_break():
    c = chunker.TextChunker(min_chars=5, max_chars=100)
    out = c.feed("First para text.\n\nSecond para.")
    out += c.flush()
    print("para ok", out)


def test_growing_stream_realistic():
    full = ("Sure! Here is a summary of the meeting. "
            "We discussed the roadmap for next quarter, including three major milestones. "
            "The first is the release of the new feature. "
            "The second is a performance improvement. "
            "The third is the onboarding flow. "
            "Overall it went well and everyone agreed on the plan. "
            "Next steps are to assign owners and set deadlines for each item. "
            "I will follow up with a detailed document tomorrow morning.")
    assert not full.endswith(".."), "test data: trailing period"
    c = chunker.TextChunker(min_chars=120, max_chars=400)
    out = []
    step = 9  # feed a few chars at a time like a real stream
    for i in range(0, len(full), step):
        out += c.feed(full[:i])
    out += c.feed(full)  # ensure the final char is fed
    out += c.flush()
    import re as _re
    rejoined = _re.sub(r"\s+", " ", " ".join(out)).strip()
    assert rejoined == _re.sub(r"\s+", " ", full).strip(), (rejoined, full)
    for ch in out:
        assert len(ch) <= 400, (len(ch), ch)
    print("realistic ok, %d chunks, total %d chars" % (len(out), sum(len(x) for x in out)))


def test_prefix_consistency_preprocessor():
    txt = "Hello **world**.\n\nThis is a [link](http://x) test. "
    c = pp.clean(txt)
    assert "world" in c and "link" in c and "**" not in c, c
    # idempotent
    assert pp.clean(c) == c
    # prefix consistency: clean(prefix) is a prefix of clean(full)
    p = txt[:10]
    assert pp.clean(txt).startswith(pp.clean(p)) or pp.clean(p) == "", (pp.clean(p), pp.clean(txt))
    print("preproc ok:", repr(c))


def test_greedy_last_paragraph_boundary():
    # p1=140 chars (a boundary >= min), p2=140. Greedy must fill toward max
    # and close at the LAST paragraph boundary under max (= end of p2), while
    # lazy closes at the FIRST (= end of p1). Same input, divergent output.
    p1 = ("Alpha " * 24).strip() + "."
    p2 = ("Beta " * 24).strip() + "."
    assert 120 <= len(p1) < 400 and 120 <= len(p2) < 400, (len(p1), len(p2))
    def stream(mode):
        text = p1 + "\n\n" + p2 + "\n\n"
        c = chunker.TextChunker(min_chars=120, max_chars=400, mode=mode)
        out = []
        for i in range(0, len(text), 9):
            out += c.feed(text[:i])
        out += c.feed(text)
        out += c.flush()
        return out

    g = stream("paragraph-greedy")
    l = stream("paragraph-lazy")
    print("greedy:", [len(x) for x in g], g)
    print("lazy:  ", [len(x) for x in l], l)
    # lazy: p1 closes at the first boundary (144 >= min 120) -> emitted alone;
    #       p2 (120 >= min) -> emitted alone.
    assert l == [p1, p2], l
    # greedy: buffers both closes (144+1+120 = 265 <= max 400) and emits them
    # combined only at flush — the chunk's close is the LAST boundary under max.
    assert g == [p1 + " " + p2], g


def test_lazy_first_boundary():
    # Covered by the divergent first-chunk assertion above; also check lazy
    # does not greedily combine when the text stops after p1.
    p1 = ("Alpha " * 24).strip() + "."
    c = chunker.TextChunker(min_chars=120, max_chars=400, mode="paragraph-lazy")
    out = []
    for i in range(0, len(p1), 7):
        out += c.feed(p1[:i])
    out += c.feed(p1)
    out += c.flush()
    print("lazy p1 only:", out)
    assert out == [p1], out


def test_tag_not_split():
    # A multi-line tag must not be split: the \n\n inside [..] is not a
    # paragraph boundary, and no chunk may end with an unclosed tag.
    c = chunker.TextChunker(min_chars=10, max_chars=400, mode="paragraph-lazy")
    text = "Hello there. " + "[whispers"
    out = []
    for i in range(0, len(text), 5):
        out += c.feed(text[:i])
    out += c.feed(text)
    tail = "\n\nand you again."
    out += c.feed(text + tail)
    out += c.flush()
    print("tag-split:", out)
    joined = " ".join(out)
    assert "[whispers" in joined, out
    # no emitted chunk ends with an unclosed tag
    for ch in out:
        opens = ch.count("[") - ch.count("]")
        assert opens <= 0, (ch, opens)


def test_greedy_sentence_fallback():
    # One long paragraph (no \n\n) of medium sentences: greedy buffers at the
    # last sentence boundary under max (104 < 100? no -> 104 > 100, so the
    # close is the last boundary under 100, i.e. end of sentence 3: 79 chars).
    c = chunker.TextChunker(min_chars=10, max_chars=100, mode="paragraph-greedy")
    out = []
    acc = ""
    for s in ["Sentence number one here. ", "Sentence number two here. ",
              "Sentence number three here. ", "Sentence number four here. "]:
        acc += s
        out += c.feed(acc)
    out += c.flush()
    print("greedy fallback:", out)
    # In-stream: closes at end-of-s3 (79 chars) -> pending 79 < 100, no emit.
    # The s4 close (26 chars) arrives after; buffer pending 104 > 100 -> emit.
    assert out, out
    joined = " ".join(out)
    assert joined == ("Sentence number one here. Sentence number two here. "
                      "Sentence number three here. Sentence number four here."), out
    for ch in out:
        assert len(ch) <= 110, (len(ch), ch)  # never far past max


def test_greedy_in_stream_emit_past_max():
    # Sentences of 50 chars each, max=100: greedy must emit in-stream once the
    # buffer exceeds max (two 50-char sentences = 101 > 100), NOT wait for the
    # end of the stream.
    c = chunker.TextChunker(min_chars=10, max_chars=100, mode="paragraph-greedy")
    full = " ".join("Sentence number %d here." % i for i in range(6)) + "."
    out = []
    for i in range(0, len(full), 7):
        out += c.feed(full[:i])
    # Mid-stream (before feeding the last sentence): already have emissions.
    assert out, ("no in-stream greedy emission", full)
    out += c.feed(full)
    out += c.flush()
    print("greedy in-stream:", [len(x) for x in out])
    assert len(out) >= 2, out


def test_oversized_sentence_slack():
    # max=400, SOFT_SLACK_FACTOR=1.5 -> slack at 600.
    def run_text(text, mode="paragraph-lazy"):
        c = chunker.TextChunker(min_chars=120, max_chars=400, mode=mode)
        out = []
        for i in range(0, len(text), 13):
            out += c.feed(text[:i])
        out += c.feed(text)
        out += c.flush()
        return text, out

    # A ~400-455 char single wordy sentence: closes at the last whitespace
    # under max (never mid-word); rejoins losslessly; all parts <= max.
    for n in (58, 65):
        text = " ".join(["word%02d" % i for i in range(n)]) + "."
        text, out = run_text(text)
        print("%d-char sentence:" % len(text), [len(x) for x in out])
        assert " ".join(out) == text.strip(), (text, out)
        for x in out:
            assert len(x) <= 400, (len(x), x)

    # A 620-char UNBROKEN sentence (no whitespace at all): past the slack the
    # char-cut fallback engages, and _split_hard re-splits word-safely. No
    # content is lost.
    text = "x" * 620 + "."
    text, out = run_text(text)
    print("620-char unbroken:", [len(x) for x in out])
    assert "".join(out) == text.strip(), (len(text), [len(x) for x in out])
    for x in out:
        assert len(x) <= 400, (len(x), x)


def test_tag_counts_toward_budget():
    # A chunk containing a big tag must count the tag text toward max.
    c = chunker.TextChunker(min_chars=10, max_chars=60, mode="paragraph-greedy")
    tag = "[a very long expression tag indeed ]"
    text = "Hi. " + tag
    out = []
    for i in range(0, len(text), 3):
        out += c.feed(text[:i])
    out += c.feed(text)
    out += c.flush()
    print("tag budget:", out)
    # 'Hi. ' (4) + tag (34) = 38 < 60, so both must sit in ONE chunk
    assert out == ["Hi. [a very long expression tag indeed ]"], out


def test_sentence_mode_unchanged():
    # The historical sentence mode must keep its exact behaviour.
    c = chunker.TextChunker(min_chars=10, max_chars=40)  # no mode -> sentence
    out = []
    acc = ""
    for s in ["Hello world. ", "How are you? ", "Fine! "]:
        acc += s
        out += c.feed(acc)
    out += c.flush()
    assert out == ["Hello world.", "How are you?", "Fine!"], out


def test_unknown_mode_raises():
    try:
        chunker.TextChunker(mode="paragraph")
    except ValueError as e:
        print("reject ok:", e)
        return
    raise AssertionError("expected ValueError for mode='paragraph'")


def test_flush_tail():
    c = chunker.TextChunker(min_chars=120, max_chars=400, mode="paragraph-lazy")
    out = c.feed("Short tail with no boundary.")
    assert out == [], out
    out += c.flush()
    assert out == ["Short tail with no boundary."], out


if __name__ == "__main__":
    test_basic_sentences()
    test_coalesce_short()
    test_long_sentence_hard_split()
    test_number_decimal_no_split()
    test_paragraph_break()
    test_growing_stream_realistic()
    test_prefix_consistency_preprocessor()
    test_greedy_last_paragraph_boundary()
    test_lazy_first_boundary()
    test_tag_not_split()
    test_greedy_sentence_fallback()
    test_oversized_sentence_slack()
    test_tag_counts_toward_budget()
    test_sentence_mode_unchanged()
    test_unknown_mode_raises()
    test_flush_tail()
    print("ALL PASS")

def test_paren_tags_atomic_default():
    # Breeze-style parentheticals are tracked tags BY DEFAULT: a
    # multi-line parenthetical is not a paragraph break, so no chunk
    # may end inside it (no dangling ')' at a chunk start).
    text = ("First line here. (long sigh\nacross a break) "
            + "word " * 40 + "End of the paragraph.\n\n"
            + "Second paragraph follows with more text. "
            + "word " * 40)
    c = chunker.TextChunker(min_chars=120, max_chars=300, mode="paragraph-lazy")
    out = []
    for i in range(0, len(text), 9):
        out += c.feed(text[:i])
    out += c.feed(text); out += c.flush()
    for k, ch in enumerate(out, 1):
        assert not ch.strip().startswith(")"), "chunk %d starts with dangling ')': %r" % (k, ch[:60])
        assert ch.count("(") == ch.count(")"), "chunk %d has unbalanced parens: %r" % (k, ch[-60:])
    print("paren-tags-default ok: %d chunks" % len(out))


def test_tag_pairs_user_config():
    # tag_pairs overrides the defaults: with "[] <>", parens are NOT tags.
    assert chunker.parse_tag_pairs(None) == chunker.DEFAULT_TAG_PAIRS
    assert chunker.parse_tag_pairs("") == chunker.DEFAULT_TAG_PAIRS
    assert chunker.parse_tag_pairs("[] <>") == (('[', ']'), ('<', '>'))
    c = chunker.TextChunker(min_chars=10, max_chars=100, mode="paragraph-lazy", tag_pairs="[] <>")
    assert c._tag_pairs == (('[', ']'), ('<', '>'))
    print("tag-pairs-config ok")


test_paren_tags_atomic_default()
test_tag_pairs_user_config()


def test_lazy_does_not_preempt_paragraphs():
    # Regression: the lazy sentence-fallback used to fire the moment the
    # buffer crossed min, BEFORE the next \n\n had arrived in the stream —
    # closing at sentence boundaries and producing ragged sub-paragraph
    # chunks. The fallback may now only fire once the buffer has passed
    # max_chars, so closes land on real paragraph boundaries whenever one
    # exists in [min, max].
    #
    # Para 1 is ~200 chars with early sentence boundaries at ~60/120: old
    # code closed at 60/120 (sub-paragraph); it must close at the \n\n.
    para1 = ("x" * 50 + ". " + "y" * 50 + ". " + "z" * 80)   # boundary at 192
    para2 = "w" * 300                                          # no sentences
    text = para1 + "\n\n" + para2
    c = chunker.TextChunker(min_chars=120, max_chars=400, mode="paragraph-lazy")
    out = []
    for i in range(0, len(text), 7):
        out += c.feed(text[:i])
    out += c.feed(text); out += c.flush()
    assert len(out) == 2, "expected 2 chunks, got %d: %s" % (len(out), [len(x) for x in out])
    # chunk 1 = whole first paragraph, NOT a 51/103-char sentence cut
    # inside it
    assert out[0].startswith("x" * 50) and out[0].endswith("z" * 80), repr(out[0][-20:])
    # chunk 2 = the whole second paragraph
    assert out[1].startswith("w")
    print("lazy-no-preempt ok: %d chunks, sizes %s" % (len(out), [len(x) for x in out]))


def test_fence_dip_does_not_reemit():
    # Regression: an open code fence's raw body sits in the clean, and the
    # closing fence collapses it to one space — the clean shrinks mid-stream.
    # feed() must rebuild the un-emitted tail, not reset and re-emit.
    import re as _re
    from collections import Counter
    paul = ("Paul\u2019s voice cut through the dead air on the speakerphone, "
            "flat and stripped of anything but utility. "
            "\u201cJohn? I\u2019m at the shelter. Tell me if you see the ground "
            "move faster than your heart rate.\u201d")
    fence = ("\n\n```python\n"
             "def guard(n):\n"
             "    return n * n if n > 3 else 0\n"
             "```\n\n"
             "The line went quiet after that. I waited.")
    full = paul + fence
    def words(s):
        return _re.findall(r"\w+", s, _re.UNICODE)
    want = Counter(words(pp.clean(full)))
    for mode in ("paragraph-lazy", "paragraph-greedy"):
        c = chunker.TextChunker(min_chars=120, max_chars=400, mode=mode)
        out = []
        for i in range(0, len(full) + 1):
            out += c.feed(pp.clean(full[:i]))
        out += c.flush()
        have = Counter(words(" ".join(out)))
        dup = {w: have[w] - want[w] for w in have if have[w] > want[w]}
        drop = {w: want[w] - have[w] for w in want if want[w] > have[w]}
        assert not dup, "mode=%s re-emitted (duplicated) words: %s" % (mode, dup)
        assert not drop, "mode=%s dropped words: %s" % (mode, drop)
        joined = " ".join(out)
        assert "guard" not in joined, "mode=%s spoke raw fence body: %r" % (mode, joined)
    print("fence-dip ok: lazy+greedy, exact-once words, no raw fence, no dup")


test_fence_dip_does_not_reemit()


test_lazy_does_not_preempt_paragraphs()
