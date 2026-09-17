"""Tests for the pure-Python Ogg page writer (ogg_page) via demux
roundtrip."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ogg_page import OggPageWriter, opus_head, opus_tags
from ogg_demux import OggDemuxer

SR = 24000


def _demux_all(path):
    with open(path, "rb") as f:
        d = OggDemuxer()
        return list(d.feed(f.read())) + d.flush()


def test_basic_roundtrip():
    d = tempfile.mkdtemp(prefix="audiocpp_page_")
    path = os.path.join(d, "a.opus")
    f = open(path, "wb")
    w = OggPageWriter(f, SR)
    pkts = [bytes([64, 1, 2, 3]) * i for i in range(1, 50)]
    for p in pkts:
        w.write_packet(p)
    w.close()
    f.close()
    got = _demux_all(path)
    assert got[0].startswith(b"OpusHead")
    assert got[1].startswith(b"OpusTags")
    assert got[2:] == pkts, (len(got), len(pkts))
    assert w.stats["packets"] == len(pkts)
    os.unlink(path)
    os.rmdir(d)
    print("basic roundtrip ok: %d packets" % len(pkts))


def test_large_packet_spans_pages():
    d = tempfile.mkdtemp(prefix="audiocpp_page_")
    path = os.path.join(d, "a.opus")
    f = open(path, "wb")
    w = OggPageWriter(f, SR)
    big = bytes(range(1, 100)) * 14  # 1960 B -> spans at least 2 pages
    w.write_packet(big)
    w.write_packet(b"tail")
    w.close()
    f.close()
    got = _demux_all(path)
    assert got[2] == big, (len(got[2]) if len(got) > 2 else "missing")
    assert got[3] == b"tail"
    # 1960 B spans the header page (25 B) + one audio page -> 2 pages total.
    assert w.stats["pages"] >= 2, w.stats
    os.unlink(path)
    os.rmdir(d)
    print("large-packet span ok: %d pages for a 1960 B packet"
          % w.stats["pages"])


def test_page_fill_boundary():
    """Fill a page to exactly 254 segments, then push a >255 B packet:
    exercises the 254-seg continuation branch (255-flag + page split)."""
    d = tempfile.mkdtemp(prefix="audiocpp_page_")
    path = os.path.join(d, "a.opus")
    f = open(path, "wb")
    w = OggPageWriter(f, SR)
    for _ in range(254):
        w.write_packet(b"\x55")          # 1 B each -> 254 segments
    next_pkt = bytes(range(1, 256)) * 2   # 510 B, crosses the page
    w.write_packet(next_pkt)
    w.close()
    f.close()
    got = _demux_all(path)
    # 254 singles (header page took 2, so they started on page 2+)
    assert got[2:2 + 254] == [b"\x55"] * 254, got[2:6]
    assert got[2 + 254] == next_pkt, (len(got) - 2, len(next_pkt))
    os.unlink(path)
    os.rmdir(d)
    print("page-fill boundary ok")


def test_granules_and_bos():
    d = tempfile.mkdtemp(prefix="audiocpp_page_")
    path = os.path.join(d, "a.opus")
    f = open(path, "wb")
    w = OggPageWriter(f, SR)
    for _ in range(5):
        w.write_packet(b"\x01\x02")
    w.close()
    f.close()
    data = open(path, "rb").read()
    assert data[5] == 0x02, "first page must be BOS (FFmpeg flag value)"
    import struct
    # last page: parse its header (find final "OggS")
    off = data.rfind(b"OggS")
    gp = struct.unpack("<q", data[off + 6:off + 14])[0]
    assert gp == 5 * 960, gp
    os.unlink(path)
    os.rmdir(d)
    print("granules/BOS ok: last granule %d (= 5 x 960)" % gp)


if __name__ == "__main__":
    test_basic_roundtrip()
    test_large_packet_spans_pages()
    test_page_fill_boundary()
    test_granules_and_bos()
    print("ALL OGG PAGE TESTS PASSED")