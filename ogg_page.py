"""Pure-Python Ogg page writer — the inverse of `ogg_demux`.

Feeds raw Opus packets (as produced by libopus `opus_encode`) into an
Ogg container, writing complete pages to a file sink as they fill.
The on-disk `.opus` therefore grows in lockstep with the live transport:
whatever the stream has delivered so far is already a valid file.

Pure `struct`/`bytes`/`zlib` — identical behaviour on Linux and Windows,
no ffmpeg, no subprocess.

Interrupted writes leave a valid prefix: pages are self-contained and
readers (browsers, ffmpeg, ogg_demux) accept a stream that ends at the
last complete page, with or without an EOS flag. We do not emit EOS
(we can't know in advance which page is last) — that is legal.

Layout implemented (RFC 3533 + Opus-in-Ogg, draft-ietf-codec-opus-in-ogg):
  page = "OggS" ver(1) htype(1) granulepos(8 LE) serial(4 LE)
         pageno(4 LE) checksum(4 LE) nsegments(1) lacing(nsegments) body
checksum = CRC-32 (poly 0x04C11DB7, MSB-first, init 0, no final XOR)
             over the page with the checksum field zeroed
  htype 0x02 (beginning-of-stream) on the very first page — FFmpeg's
  de facto convention (its oggdec.h defines OGG_FLAG_BOS as 2 and its
  muxer writes 2); RFC 3533's 0x01 is not checked by FFmpeg.
  OpusHead on the BOS page, OpusTags on its own page (no audio); granule 0.
  Subsequent page granule = (audio packets through end of page) * 960
  (Opus counts time in 48 kHz units: 20 ms frame = 960).
  A packet spans consecutive segments (255 B max each); a packet crossing
  a page boundary has 0xFF lacing on the final segment of the earlier page.
"""

import os
import struct

MAGIC = b"OggS"
BOS = 0x02  # FFmpeg's OGG_FLAG_BOS; the RFC's 0x01 is not checked by FFmpeg
# libogg (oggpcrc) checksum table: polynomial 0x04C11DB7, MSB-first,
# init 0, no final XOR. This is the Ogg page checksum of RFC 3533 —
# NOT zlib.crc32 (reflected polynomial + final XOR) and NOT adler32.
_CRC_POLY = 0x04C11DB7


def _build_crc_table():
    table = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = ((r << 1) ^ _CRC_POLY) if (r & 0x80000000) else (r << 1)
            r &= 0xFFFFFFFF
        table.append(r)
    return tuple(table)


_CRC_TABLE = _build_crc_table()


def ogg_page_crc(data):
    """Checksum of `data` per the Ogg page rules (libogg oggpcrc)."""
    crc = 0
    for b in data:
        crc = _CRC_TABLE[(crc >> 24) ^ b] ^ (crc << 8)
        crc &= 0xFFFFFFFF
    return crc
def opus_head(sample_rate, channels=1):
    # version(1) channels(1) pre_skip(2 LE) input_sample_rate(4 LE)
    # output_gain(2 LE) mapping_family(1) — 19 bytes, RFC 7845
    return (b"OpusHead" + bytes([1, channels, 0, 0])
            + struct.pack("<I", int(sample_rate)) + b"\x00\x00"
            + bytes([0]))


def opus_tags(vendor=b"audio_cpp"):
    return (b"OpusTags" + struct.pack("<I", len(vendor)) + vendor
            + struct.pack("<I", 0))


class OggPageWriter:
    """Streams raw Opus packets into an Ogg file sink.

    `sink` is anything with `.write(bytes)`. The writer creates the file's
    OpusHead/OpusTags page itself (in `start`'s caller: `__init__`).
    """

    def __init__(self, sink, sample_rate=24000, channels=1,
                 vendor=b"audio_cpp", serial=None):
        self.sink = sink
        self.sample_rate = int(sample_rate)
        if serial is None:
            serial = struct.unpack("<I", os.urandom(4))[0]
        self.serial = int(serial)
        self._seq = 0
        self._granule = 0
        self._body = bytearray()
        self._seglen = []
        self._pages = 0
        self._packets = 0
        self._begin = True
        # Two separate header pages, granule 0: OpusHead on the BOS page,
        # OpusTags on its own. FFmpeg's opus header state machine expects
        # the tags packet via its header() callback (need_comments); a
        # tags packet squashed onto the BOS page is consumed by the
        # packet() path instead and the expectation is never cleared.
        self._write_packets_page([opus_head(sample_rate, channels)])
        self._write_packets_page([opus_tags(vendor)])

    # -- page emission -----------------------------------------------------
    def _emit_page(self, body, seglen, granule):
        if not seglen:
            return
        n = len(seglen)
        htype = BOS if self._begin else 0
        # 26-byte fixed header, checksum slot zeroed, then
        # seg-count + lacing. The checksum covers the WHOLE page
        # (26 fixed + seg-count + lacing + body) with the 4 checksum
        # bytes zeroed, spliced back in — RFC 3533 §4.4.1.
        fixed = (MAGIC + b"\x00" + bytes([htype])
                 + struct.pack("<qII", granule, self.serial, self._seq)
                 + b"\x00\x00\x00\x00")  # 26 bytes, checksum slot zeroed
        rest = bytes([n]) + bytes(seglen) + bytes(body)
        cs = ogg_page_crc(fixed + rest)
        self.sink.write(fixed[:22] + struct.pack("<I", cs) + fixed[26:] + rest)
        self._seq += 1
        self._pages += 1
        self._begin = False

    def _write_packets_page(self, packets):
        """Write a page containing exactly `packets` (total <= 255 segs)."""
        body = bytearray()
        seglen = []
        for p in packets:
            off = 0
            while off < len(p):
                take = min(255, len(p) - off)
                seglen.append(take)
                body.extend(p[off:off + take])
                off += take
        self._emit_page(body, seglen, self._granule)

    # -- streaming packets ---------------------------------------------------
    def write_packet(self, pkt):
        """One raw Opus audio packet; may span multiple pages."""
        pkt = bytes(pkt)
        lacing = self._lacing(pkt)
        self._packets += 1
        # RFC 7845: granules are in 48 kHz units regardless of the stream's
        # sample rate — one 20 ms Opus frame is always 960.
        self._granule += 960
        for i in range(0, len(lacing), 255):
            chunk = lacing[i:i + 255]
            body = bytearray()
            seglen = []
            for size in chunk:
                seglen.append(size)
                body.extend(pkt[:size])
                pkt = pkt[size:]
            self._emit_page(bytes(body), seglen, self._granule)

    def _lacing(self, pkt):
        """RFC 3533 lacing for one whole packet.

        Runs of 255 mean "continues"; the final 255-byte run is written as
        255 then 0 so it is NOT mistaken for a cross-page continuation.
        """
        lacing = []
        off = 0
        while off < len(pkt):
            rem = len(pkt) - off
            if rem > 255:
                lacing.append(255)
                off += 255
            else:
                lacing.append(rem)
                off += rem
        if lacing and lacing[-1] == 255:
            lacing.append(0)
        return lacing

    def _flush_page(self):
        self._emit_page(self._body, self._seglen, self._granule)
        self._body = bytearray()
        self._seglen = []

    def close(self):
        """Flush a trailing partial page. (No EOS flag — legal to omit.)"""
        if self._seglen:
            self._flush_page()

    @property
    def stats(self):
        return {"pages": self._pages, "packets": self._packets,
                "granule": self._granule}