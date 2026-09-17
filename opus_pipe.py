"""In-process streaming PCM -> Ogg/Opus encoder (one per voice session).

Replaces the ffmpeg subprocess pipe, whose ogg muxer buffered ALL stdout
until EOF (measured) and therefore could not deliver live packets.

     PCM16-LE mono @ 24 kHz  --feed()-->  libopus (ctypes)  -->
         (a) raw Opus packets returned for the live SSE transport, and
         (b) the same packets wrapped into Ogg pages by `ogg_page` and
             appended to the session's .opus file, in lockstep.

The file IS the streamed output: no post-hoc batch encode. An
interrupted session leaves a valid Ogg/Opus prefix on disk (pages are
independent; readers accept a stream cut at the last complete page).

Cross-platform (Linux + Windows): libopus is loaded by a name list
(opus / libopus.so.0 / libopus-0.dll ...), overridable via the
`AUDIOCPP_LIBOPUS` environment variable (full path). Missing library ->
`load_libopus()` raises `RuntimeError` and the relay falls back to the
PCM transport. No ffmpeg anywhere in this path.

Performance (measured): one 20 ms frame encodes in ~0.16 ms at
24 kHz mono 64 kbps — ~1200x real time. `feed()` does the work
synchronously on the calling (worker) thread: no reader thread, no
lifecycle races.
"""

import ctypes
import ctypes.util
import os
import threading
import traceback

try:
    from . import ogg_page
except ImportError:  # direct-script execution (tests)
    import ogg_page

_log = None


def set_logger(fn):
    global _log
    _log = fn


def _dbg(msg):
    if _log is not None:
        try:
            _log(msg)
        except Exception:
            pass


# -- libopus binding -------------------------------------------------------
_LIB_NAMES = [
    "opus",            # Windows: resolves to opus.dll
    "libopus.so.0",    # Debian/Ubuntu
    "libopus.so",      # others
    "libopus-0.dll",   # ffmpeg-bundled copy (WSL)
]


def load_libopus():
    """Locate and bind libopus. Returns the ctypes lib, else raises
    RuntimeError with a diagnostic string."""
    override = os.environ.get("AUDIOCPP_LIBOPUS", "").strip()
    names = [override] if override else _LIB_NAMES
    for name in names:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        try:
            fn = getattr(lib, "opus_encoder_create")
        except AttributeError:
            continue
        _configure(lib)
        _dbg("[opus_pipe] bound libopus: %s (%s)"
             % (name, libopus_version(lib)))
        return lib
    raise RuntimeError(
        "libopus not found (tried: %s); set AUDIOCPP_LIBOPUS to a full "
        "path to the shared library" % ", ".join(names))


def _configure(lib):
    lib.opus_get_version_string.restype = ctypes.c_char_p
    lib.opus_encoder_create.restype = ctypes.c_void_p
    lib.opus_encoder_create.argtypes = [ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.opus_encoder_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_int64]
    lib.opus_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16),
                                ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.opus_encode.restype = ctypes.c_int
    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]


def libopus_version(lib):
    try:
        return lib.opus_get_version_string().decode("utf-8", "replace")
    except Exception:
        return "?"


# libopus constants
APP_AUDIO = 2048                 # OpusApplication.OPUS_APPLICATION_AUDIO
CTL_SET_BITRATE = 4002           # OPUS_SET_BITRATE
MAX_FRAME = 960                  # 20 ms @ 48 kHz (max frame size, in samples)
MAX_PACKET = 1275                # RFC 6716 max payload
OPUS_OK = 0

_lib = None
_lib_lock = threading.Lock()


def _get_lib():
    global _lib
    if _lib is None:
        with _lib_lock:
            if _lib is None:
                _lib = load_libopus()
    return _lib


# -- the pipe -----------------------------------------------------------------
class OpusEncoderPipe:
    """Feed PCM; get raw Opus packets; the .opus file grows in lockstep.

    Interface is unchanged from the ffmpeg-pipe version:
      start() -> self; feed(pcm) -> [raw packets]; drain() -> [packets]
      finish() -> (packets, path_or_None); kill() -> (partial file kept)
    """

    def __init__(self, dest, sample_rate, bitrate=64000, channels=1,
                 vendor=b"audio_cpp"):
        self.dest = dest
        self.sample_rate = int(sample_rate)
        self.bitrate = int(bitrate)
        self.channels = int(channels)
        self.vendor = vendor
        self._lib = None
        self._enc = None
        self._file = None
        self._writer = None
        self._pending = []
        self._lock = threading.Lock()
        self._feed_bytes = 0
        self._frames = 0
        self._closed = False
        self._carry = bytearray()   # trailing partial-frame PCM, < one frame

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        self._lib = _get_lib()
        err = ctypes.c_int(0)
        enc = self._lib.opus_encoder_create(self.sample_rate, self.channels,
                                            APP_AUDIO, ctypes.byref(err))
        if not enc or err.value != OPUS_OK:
            raise RuntimeError("opus_encoder_create failed: %d" % err.value)
        self._lib.opus_encoder_ctl(enc, CTL_SET_BITRATE, self.bitrate)
        os.makedirs(os.path.dirname(self.dest), exist_ok=True)
        self._file = open(self.dest, "wb")
        self._writer = ogg_page.OggPageWriter(self._file, self.sample_rate,
                                              self.channels, self.vendor)
        self._enc = enc
        _dbg("[opus_pipe] started -> %s (sr=%d ch=%d br=%d libopus=%s)"
             % (self.dest, self.sample_rate, self.channels, self.bitrate,
                libopus_version(self._lib)))
        return self

    # -- data path ----------------------------------------------------------
    def feed(self, pcm):
        """Encode `pcm` (16-bit LE, `self.channels` channels,
        `self.sample_rate` Hz). Returns the Opus packets produced (plus any
        backlog). Trailing partial frame is carried to the next feed."""
        if self._enc is None:
            return []
        # frame = 20 ms at the encoder's own rate (960 samples @ 48 kHz
        # is libopus's reference; at 24 kHz it is 480 samples)
        frame_samples = 960 * self.sample_rate // 48000
        frame_bytes = frame_samples * 2 * self.channels
        # Carry the previous feed's trailing partial frame forward.
        pcm = bytes(self._carry) + bytes(pcm)
        self._carry = pcm[(len(pcm) // frame_bytes) * frame_bytes:]
        n = len(pcm) // frame_bytes
        if n == 0:
            return self.drain()
        samples = n * frame_samples * self.channels
        buf = (ctypes.c_int16 * samples)()
        ctypes.memmove(ctypes.addressof(buf), pcm, samples * 2)
        buf_base = ctypes.addressof(buf)
        pkts = []
        for i in range(n):
            frame = ctypes.cast(buf_base + i * frame_bytes,
                                 ctypes.POINTER(ctypes.c_int16))
            out = ctypes.create_string_buffer(MAX_PACKET)
            r = self._lib.opus_encode(self._enc, frame, frame_samples, out,
                                       MAX_PACKET)
            if r < 0:
                _dbg("[opus_pipe] opus_encode failed: %d (frame %d)"
                     % (r, self._frames + i))
                continue
            pkt = out.raw[:r]
            self._writer.write_packet(pkt)
            self._flush_file()
            pkts.append(pkt)
        self._feed_bytes += n * frame_bytes
        self._frames += n
        with self._lock:
            self._pending.extend(pkts)
        return self.drain()

    def _flush_file(self):
        try:
            self._file.flush()
        except Exception:
            pass

    def drain(self):
        with self._lock:
            pkts, self._pending = self._pending, []
        return pkts

    # -- teardown -----------------------------------------------------------
    def finish(self, timeout=30.0):
        """Close the stream. Returns (packets, file_path_or_None); the file
        is a complete, playable Ogg/Opus."""
        pkts = self._close_stream(pad=True)
        ok = os.path.exists(self.dest) and os.path.getsize(self.dest) > 0
        _dbg("[opus_pipe] finished pcm_in=%dB frames=%d ogg_out=%dB packets=%d"
             % (self._feed_bytes, self._frames,
                os.path.getsize(self.dest) if ok else -1, len(pkts)))
        return pkts, (self.dest if ok else None)

    def kill(self):
        """HARD stop. The partial file remains a valid (prefix) Ogg/Opus
        stream; callers may keep or delete it."""
        self._close_stream()
        _dbg("[opus_pipe] killed (hard stop); partial file kept: %s"
             % self.dest)

    def _close_stream(self, pad=False):
        if self._enc is not None and self._carry:
            # Encode the trailing partial frame, zero-padded to frame size.
            frame_samples = 960 * self.sample_rate // 48000
            frame_bytes = frame_samples * 2 * self.channels
            padded = bytes(self._carry) + b"\x00\x00" * (
                frame_bytes - len(self._carry))
            self._carry = bytearray()
            buf = (ctypes.c_int16 * frame_samples)()
            ctypes.memmove(ctypes.addressof(buf), padded,
                           frame_bytes * self.channels)
            out = ctypes.create_string_buffer(MAX_PACKET)
            r = self._lib.opus_encode(self._enc, buf, frame_samples, out,
                                       MAX_PACKET)
            if r > 0 and pad:
                pkt = out.raw[:r]
                self._writer.write_packet(pkt)
                self._flush_file()
                with self._lock:
                    self._pending.append(pkt)
        pkts = self.drain()
        if not self._closed:
            self._closed = True
            try:
                if self._writer is not None:
                    self._writer.close()
                if self._file is not None:
                    self._file.flush()
                    self._file.close()
                if self._enc is not None and self._lib is not None:
                    self._lib.opus_encoder_destroy(self._enc)
            except Exception:
                traceback.print_exc()
            self._writer = None
            self._file = None
            self._enc = None
        return pkts