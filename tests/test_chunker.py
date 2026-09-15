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


if __name__ == "__main__":
    test_basic_sentences()
    test_coalesce_short()
    test_long_sentence_hard_split()
    test_number_decimal_no_split()
    test_paragraph_break()
    test_growing_stream_realistic()
    test_prefix_consistency_preprocessor()
    print("ALL PASS")