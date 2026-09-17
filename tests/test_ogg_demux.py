"""Tests for the Ogg/Opus demuxer.

Run: python3 tests/test_ogg_demux.py
Plain-assertion style (no unittest), matching the other audio_cpp tests.
"""
import os
import random
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ogg_demux import OggDemuxer, _decode_packets


def _make_page(lacing, body):
    """Build a structurally-correct Ogg page (fixed 26-byte header).

    Layout: 4 magic + 1 version + 1 header-type + 8 granule + 4 serial
    + 4 page-seq + 4 checksum = 26 fixed bytes, then 1 seg-count byte
    (offset 26), then the lacing table (offset 27), then the body.
    """
    assert sum(lacing) == len(body), (lacing, len(body))
    hdr = (b"OggS" + b"\x00"  # version 0
           + b"\x02"          # header-type: continuation of stream
           + b"\x00" * 8      # granule (placeholder)
           + b"\x00" * 4      # bitstream serial
           + b"\x00" * 4      # page sequence
           + b"\x00" * 4)     # checksum (not validated by demuxer)
    hdr += bytes([len(lacing)])
    return hdr + bytes(lacing) + body


def test_decode_packets_synthetic():
    body = bytes(range(256)) + bytes(17)
    # [10, 255, 5, 3]: pkt A = 10 bytes (lacing 10 < 255 closes it);
    # pkt B = 255+5 = 260 bytes (255 continues, 5 closes); pkt C = 3 bytes.
    pk = _decode_packets([10, 255, 5, 3], body)
    assert len(pk) == 3, len(pk)
    assert pk[0] == body[0:10]
    assert pk[1] == body[10:270]
    assert pk[2] == body[270:273]
    # Single-segment single-packet page.
    assert _decode_packets([7], bytes(7)) == [bytes(7)]
    # Two independent packets in one page.
    b10 = bytes(range(10))
    assert _decode_packets([4, 6], b10) == [b10[0:4], b10[4:10]]
    # Page ending mid-packet (all-255 lacing): no complete packet yet.
    assert _decode_packets([255, 255], bytes(510)) == []


def test_buffering_split_across_feeds():
    # One real page: 2 segments (3-byte + 4-byte packets).
    body = bytes([1, 2, 3, 4, 5, 6, 7])
    page = _make_page([3, 4], body)
    assert len(page) == 26 + 1 + 2 + 7, len(page)
    d = OggDemuxer()
    # Feed the page 1 byte at a time.
    pk = []
    for i in range(len(page)):
        pk.extend(d.feed(page[i:i + 1]))
    assert pk == [b"\x01\x02\x03", b"\x04\x05\x06\x07"], pk


def test_ffmpeg_roundtrip():
    pcm = (b"\x01\x00" * 24000 * 5)  # 5 s, 24 kHz mono s16le
    ffmpeg = os.environ.get("AUDIOCPP_FFMPEG", "ffmpeg")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error",
           "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "-",
           "-c:a", "libopus", "-b:a", "64k", "-f", "ogg", "-"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, _ = p.communicate(input=pcm, timeout=60)
    assert p.returncode == 0, "ffmpeg failed"
    assert out.startswith(b"OggS"), "not an ogg stream"

    d = OggDemuxer()
    random.seed(1)
    pkts, i, n = [], 0, len(out)
    while i < n:
        c = min(random.randrange(1, 4000), n - i)
        pkts.extend(d.feed(out[i:i + c]))
        i += c
    pkts.extend(d.flush())

    assert len(pkts) >= 2, "expected header packets"
    head, tags = pkts[0], pkts[1]
    assert head.startswith(b"OpusHead"), head[:8]
    assert tags.startswith(b"OpusTags"), tags[:8]
    # Verify the header declares 24 kHz mono (no resample happened).
    channels = head[9]
    sr = struct.unpack("<I", head[12:16])[0]
    assert (channels, sr) == (1, 24000), (channels, sr)

    audio = pkts[2:]
    # 5 s of audio at 20 ms frames = ~250 packets (allow encoder flush drift).
    assert 240 <= len(audio) <= 260, len(audio)
    print("roundtrip ok: %d pages -> %d packets (%d audio, %d bytes ogg)"
          % (len(out) // 27, len(pkts), len(audio), len(out)))


if __name__ == "__main__":
    test_decode_packets_synthetic()
    test_buffering_split_across_feeds()
    test_ffmpeg_roundtrip()
    print("all ogg demux tests ok")