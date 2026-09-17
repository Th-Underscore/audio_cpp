"""Minimal Ogg page demux for Opus packets (RFC 3533 + RFC 7845).

Splits a raw Ogg/Opus byte stream (e.g. ffmpeg's `-f ogg` stdout) into the
individual Opus packets, in order. Each 20 ms TTS frame arrives in its own
page, so the simple single-lacing-segment-per-page reading covers the
streaming case; multi-lacing pages are also handled by the standard
lacing-pattern decode (the W3C WebCodecs OggParser uses the same rules).

Used to turn a persistent encoder's stdout into per-packet SSE events for
the browser's raw-Opus WebCodecs path — the same bytes are tee'd to the
finished .opus file, so no second encode pass exists.

Pure functions; no I/O.

Page layout (RFC 3533 §4), all offsets in bytes:
    0   "OggS" capture pattern (4)
    4   version (1)
    5   header type flags (1)
    6   granule position (8)
    14  bitstream serial number (4)
    18  page sequence number (4)
    22  checksum (4)
    26  number of physical segments (1)
    27  lacing values, one byte per segment
    27+n  segment data
So the fixed header is 26 bytes, seg_count is at offset 26, and the lacing
table begins at offset 27.
"""

OGG_MAGIC = b"OggS"
OGG_SEG_COUNT_OFF = 26
OGG_LACING_OFF = 27
# Smallest possible page: 26-byte fixed header + 1 lacing byte.
OGG_MIN_PAGE = 27


class OggDemuxer:
    """Stateful splitter: feed() chunks of the stream, get packets back."""

    def __init__(self):
        self._buf = bytearray()
        self._partial = bytearray()  # packet continuing across a page break

    def feed(self, data: bytes) -> list:
        """Consume a chunk of the byte stream; returns [bytes] packets."""
        self._buf.extend(data)
        packets = []
        while len(self._buf) >= OGG_MIN_PAGE:
            off = self._buf.find(OGG_MAGIC)
            if off < 0:
                # No sync in the buffer: if it's a long run of non-Ogg
                # bytes the real sync may already be gone; keep only the
                # last 3 (max prefix of "OggS" that could start a page).
                if len(self._buf) > 3:
                    del self._buf[0:len(self._buf) - 3]
                break
            if off > 0:
                del self._buf[0:off]  # drop inter-page garbage
            seg_count = self._buf[OGG_SEG_COUNT_OFF]
            # Need the whole lacing table before we know the page length.
            if len(self._buf) < OGG_LACING_OFF + seg_count:
                break  # incomplete page; wait for more bytes
            # Lacing value is a full 8-bit field: 255 == packet continues,
            # 0-254 == segment length. Do NOT mask — a masked value (e.g.
            # 160 & 0x1F == 0) truncates real packets to zero bytes.
            sizes = [self._buf[OGG_LACING_OFF + i]
                      for i in range(seg_count)]
            body_off = OGG_LACING_OFF + seg_count
            body_len = sum(sizes)
            # The page is complete only when its full body has arrived.
            if len(self._buf) < body_off + body_len:
                break  # incomplete page body; wait for more bytes
            body = bytes(self._buf[body_off:body_off + body_len])
            del self._buf[0:body_off + body_len]
            # A page may end mid-packet (last lacing == 255): carry that
            # partial packet into the next page's first segments. Seed
            # with the carried-over partial (empty if none); each
            # completed packet then starts from a FRESH buffer — reusing
            # one object here makes later packets accumulate onto it.
            cur = bytearray(self._partial)
            self._partial = bytearray()
            pos = 0
            for size in sizes:
                cur.extend(body[pos:pos + size])
                pos += size
                if size != 255:
                    packets.append(bytes(cur))
                    cur = bytearray()
            if cur:
                self._partial = cur
        return packets

    def flush(self) -> list:
        """End of stream: discard incomplete trailing bytes.

        ffmpeg's encoder flush is written before the process exits, so a
        well-formed stream is complete by EOF; anything left here is a
        truncated tail page and is unusable.
        """
        self._buf.clear()
        return []


def _decode_packets(lacing: list, body: bytes) -> list:
    """Decode ONE page's lacing into packets, assuming the page starts and
    ends at packet boundaries (no cross-page continuation).

    Kept for callers/tests that decode a single page in isolation; the
    streaming demuxer's feed() has its own cross-page-aware variant.
    """
    packets = []
    cur = None
    pos = 0
    for size in lacing:
        if cur is None:
            cur = bytearray()
        cur.extend(body[pos:pos + size])
        pos += size
        if size != 255:
            packets.append(bytes(cur))
            cur = None
    return packets