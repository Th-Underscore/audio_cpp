"""Tests for the in-process opus encoder pipe (opus_pipe + ogg_page)."""
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ogg_demux import OggDemuxer
from opus_pipe import OpusEncoderPipe, load_libopus, libopus_version

SR = 24000


def _pcm(seconds):
    return b"\x01\x02" * int(SR * seconds)


def test_load():
    lib = load_libopus()
    print("libopus:", libopus_version(lib))


def test_roundtrip():
    d = tempfile.mkdtemp(prefix="audiocpp_pipe_")
    dest = os.path.join(d, "t.opus")
    p = OpusEncoderPipe(dest, SR).start()
    pk = list(p.feed(_pcm(5)))
    fin_pkts, path = p.finish()
    assert path == dest, path
    pk += fin_pkts
    total = len(pk)
    assert total >= 240, "expected ~250 packets for 5s, got %d" % total
    file_size = os.path.getsize(dest)
    # The file re-demuxes independently and yields the SAME packets.
    with open(dest, "rb") as f:
        d2 = OggDemuxer()
        file_pkts = list(d2.feed(f.read())) + d2.flush()
    assert len(file_pkts) == total + 2, (len(file_pkts), total)  # +OpusHead/Tags
    assert file_pkts[2:] == pk, "file packets != streamed packets"
    assert file_pkts[0].startswith(b"OpusHead")
    assert file_pkts[1].startswith(b"OpusTags")
    audio_bytes = sum(len(x) for x in file_pkts[2:])
    kbps = audio_bytes * 8 / 5 / 1000
    print("roundtrip ok: %d packets, file %d bytes, ~%.1f kbps" %
          (total, file_size, kbps))
    shutil.rmtree(d, ignore_errors=True)


def test_incremental_feeds_accumulate():
    d = tempfile.mkdtemp(prefix="audiocpp_pipe_")
    dest = os.path.join(d, "t.opus")
    p = OpusEncoderPipe(dest, SR).start()
    got = 0
    for _ in range(50):
        got += len(p.feed(_pcm(0.1)))
    got += len(p.finish()[0])
    assert got >= 240, "expected ~250, got %d" % got
    print("incremental ok: %d packets" % got)
    shutil.rmtree(d, ignore_errors=True)


def test_partial_frame_carry():
    # 20 ms at 24 kHz = 480 samples per frame (frame size is rate-dependent;
    # only 48000 Hz uses 960).
    frame = int(SR * 0.020)
    d = tempfile.mkdtemp(prefix="audiocpp_pipe_")
    dest = os.path.join(d, "t.opus")
    p = OpusEncoderPipe(dest, SR).start()
    # Exactly half a frame: no packet yet.
    half = p.feed(b"\x01\x02" * (frame // 2))
    assert half == [], half
    # Finish the frame plus one more: exactly 2 packets.
    rest = p.feed(b"\x01\x02" * (frame // 2 + frame))
    assert len(rest) == 2, len(rest)
    p.kill()
    print("partial-frame carry ok")
    shutil.rmtree(d, ignore_errors=True)


def test_kill_leaves_valid_prefix():
    d = tempfile.mkdtemp(prefix="audiocpp_pipe_")
    dest = os.path.join(d, "t.opus")
    p = OpusEncoderPipe(dest, SR).start()
    pk_live = list(p.feed(_pcm(2)))
    assert len(pk_live) >= 100, len(pk_live)
    p.kill()
    size = os.path.getsize(dest)
    assert size > 0, "killed pipe left no file"
    with open(dest, "rb") as f:
        d2 = OggDemuxer()
        file_pkts = list(d2.feed(f.read())) + d2.flush()
    # File = header + exactly the packets we streamed so far.
    assert file_pkts[0].startswith(b"OpusHead")
    assert file_pkts[2:] == pk_live, (len(file_pkts), len(pk_live))
    print("kill-prefix ok: %d packets, %d bytes, re-demuxes cleanly"
          % (len(file_pkts), size))
    shutil.rmtree(d, ignore_errors=True)


def test_deterministic_split():
    # Feeding 5s as one chunk vs 25 x 0.2s must yield identical packets.
    d = tempfile.mkdtemp(prefix="audiocpp_pipe_")
    dest1, dest2 = os.path.join(d, "a.opus"), os.path.join(d, "b.opus")
    p1 = OpusEncoderPipe(dest1, SR).start()
    one = list(p1.feed(_pcm(5)))
    p1.kill()
    p2 = OpusEncoderPipe(dest2, SR).start()
    two = []
    for _ in range(25):
        two += p2.feed(_pcm(0.2))
    p2.kill()
    assert one == two, (len(one), len(two))
    print("deterministic split ok: %d packets" % len(one))
    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_load()
    test_roundtrip()
    test_incremental_feeds_accumulate()
    test_partial_frame_carry()
    test_kill_leaves_valid_prefix()
    test_deterministic_split()
    print("ALL OPUS PIPE TESTS PASSED")