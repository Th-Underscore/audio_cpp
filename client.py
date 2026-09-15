"""Thin HTTP client for an audio.cpp server.

Only textgen talks to audio.cpp — the browser never does. This module wraps
the small surface we need:

  GET  /v1/models
  GET  /v1/audio/voices?model=<id>
  POST /v1/audio/speech   (stream_format: sse, response_format: pcm)

The TTS call is a generator yielding decoded PCM16 bytes as audio.cpp streams
them (each `speech.audio.delta` SSE event carries base64 PCM16-LE mono).

No audio.cpp-specific assumptions beyond the verified event shape
(app/server/runtime.cpp:2250-2254). `input` is the text field (NOT `text`).
"""

import base64
import json
import threading
import time
import urllib.parse
import urllib.request
import urllib.error

DEFAULT_TIMEOUT = 120  # seconds for a single TTS request

# Optional debug hook: set via set_logger(callable) to capture request/response
# detail. Kept out of the client itself so it stays textgen-independent; the
# extension wires it to shared.logger in setup().
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


def _preview(text, limit=200):
    """Truncate a string for logging with a VISIBLE truncation marker."""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + " ...[+%d more chars, total %d]" % (len(text) - limit, len(text))


def _req(url, body=None, headers=None, timeout=DEFAULT_TIMEOUT, method=None):
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    headers = dict(headers or {})
    if body is not None:
        headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def get_models(server_url, timeout=10):
    """Return list of (id, family) from GET /v1/models (family may be '')."""
    from . import specs
    with _req(server_url.rstrip("/") + "/v1/models", timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    items = payload.get("data", payload) if isinstance(payload, dict) else payload
    out = []
    for m in items or []:
        if isinstance(m, dict) and m.get("id"):
            fam = (m.get("family") or "").strip()
            out.append((m["id"], fam or specs.detect_family(m["id"]) or ""))
    _dbg("[models] GET %s/v1/models -> %s" % (server_url, out))
    return out


def get_model_families(models):
    """{id: family} from get_models() output (tolerates bare-id lists)."""
    fams = {}
    for m in models or []:
        if isinstance(m, (tuple, list)) and len(m) == 2:
            fams[m[0]] = m[1] or ""
        elif isinstance(m, str):
            from . import specs
            fams[m] = specs.detect_family(m) or ""
    return fams


def get_voices(server_url, model=None, timeout=10):
    """Return list of voice names from GET /v1/audio/voices?model=<id>."""
    url = server_url.rstrip("/") + "/v1/audio/voices"
    if model:
        url += "?model=" + urllib.parse.quote(model)
    try:
        with _req(url, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        _dbg("[voices] GET %s failed: %r" % (url, e))
        return []
    _dbg("[voices] GET %s raw=%s" % (url, _preview(json.dumps(payload), 300)))
    # audio.cpp shape: {"voices": ["name", ...]} (runtime.cpp handle_voices).
    # Tolerate an OpenAI-style {"data": [{"id": ...}]} too.
    if isinstance(payload, dict):
        items = payload.get("voices", payload.get("data", []))
    else:
        items = payload
    out = []
    for v in items or []:
        if isinstance(v, dict):
            name = v.get("id") or v.get("name")
            if name:
                out.append(name)
        elif isinstance(v, str):
            out.append(v)
    return out


def stream_tts(server_url, model, text, options=None, voice=None,
                response_format="pcm", stream_format="sse", seed=None,
                timeout=DEFAULT_TIMEOUT, cancel_event=None):
    """POST /v1/audio/speech and yield decoded PCM16 bytes per delta event.

    `options` is the audio.cpp `options` object (instruction, temperature,
    top_p, text_chunk_mode, text_chunk_size, stream_frames_per_event, ...).
    Yields raw PCM16-LE mono bytes. Stops at `speech.audio.done` or `[DONE]`.
    Raises on a non-200 HTTP status or an `error` SSE event.

    `cancel_event` (threading.Event) — when set, the generator stops after
    the in-flight read (connection closed) WITHOUT raising. Used by the
    stop/interrupt button: the audio.cpp request is dropped mid-stream.
    """
    body = {
        "model": model,
        "input": text,
        "response_format": response_format,
        "stream": True,
        "stream_format": stream_format,
    }
    if voice:
        body["voice"] = voice
    if seed is not None:
        body["seed"] = seed
    if options:
        from . import specs
        kept, dropped, fam = specs.filter_options(model, options)
        if dropped:
            _dbg("[tts] spec filter (family=%s): DROPPED unknown options %s"
                 % (fam, dropped))
        body["options"] = kept

    _dbg("[tts] POST %s/v1/audio/speech body=%s"
         % (server_url, json.dumps({
             "model": body.get("model"),
             "input": _preview(body.get("input")),
             "response_format": body.get("response_format"),
             "stream": body.get("stream"),
             "stream_format": body.get("stream_format"),
             "voice": body.get("voice"),
             "seed": body.get("seed"),
             "options": body.get("options"),
         })))
    n_delta = 0
    n_pcm = 0
    t0 = time.time()
    try:
        if cancel_event is not None and cancel_event.is_set():
            _dbg("[tts] cancelled before request")
            return
        with _req(server_url.rstrip("/") + "/v1/audio/speech",
                  body=body, timeout=timeout) as r:
            _dbg("[tts] HTTP %s %s ctype=%s len=%s"
                 % (r.status, getattr(r, "reason", ""),
                    r.headers.get("Content-Type"),
                    r.headers.get("Content-Length")))
            # Read SSE line-by-line from the response stream.
            buf = b""
            for raw_line in _iter_sse_lines(r):
                line = raw_line.strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    _dbg("[tts] non-data SSE line: %r" % line[:160])
                    continue
                payload_str = line[len("data:"):].strip()
                if payload_str == "[DONE]":
                    _dbg("[tts] [DONE] after %d deltas / %dB pcm"
                         % (n_delta, n_pcm))
                    return
                try:
                    ev = json.loads(payload_str)
                except json.JSONDecodeError:
                    _dbg("[tts] unparseable SSE payload: %r" % _preview(payload_str))
                    continue
                etype = ev.get("type")
                if etype == "speech.audio.delta":
                    b64 = ev.get("audio")
                    if not b64:
                        _dbg("[tts] delta with no audio field: %r"
                             % _preview(json.dumps(ev)))
                        continue
                    pcm = base64.b64decode(b64)
                    n_delta += 1
                    n_pcm += len(pcm)
                    if n_delta == 1:
                        _dbg("[tts] first delta: %dB pcm" % len(pcm))
                    yield pcm
                    if cancel_event is not None and cancel_event.is_set():
                        _dbg("[tts] CANCELLED after %d deltas / %dB pcm "
                             "(%.1fs)" % (n_delta, n_pcm, time.time() - t0))
                        return
                elif etype == "speech.audio.done":
                    _dbg("[tts] speech.audio.done after %d deltas / %dB pcm "
                         "(%.1fs)" % (n_delta, n_pcm, time.time() - t0))
                    return
                elif etype == "error":
                    _dbg("[tts] error event: %s" % json.dumps(ev.get("error")))
                    raise RuntimeError("audio.cpp TTS error: %s" % json.dumps(ev.get("error")))
            _dbg("[tts] stream ended without done/[DONE] after %d deltas / %dB pcm"
                 % (n_delta, n_pcm))
    except Exception as e:
        _dbg("[tts] EXCEPTION after %d deltas / %dB pcm: %r" % (n_delta, n_pcm, e))
        raise


def _iter_sse_lines(response):
    """Yield individual text lines from a streaming HTTP response body."""
    buf = b""
    while True:
        chunk = response.read(1024)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode("utf-8", "replace")
    if buf:
        yield buf.decode("utf-8", "replace")

class VoiceDiscovery:
    """Auto-discover audio.cpp models/voices; retries forever with exponential backoff.

    Each endpoint delegates to this: e.g.
        models = voice_discov.models_cache if voice_discov else []

    `discover_and_cache()` runs a background thread (stops on shutdown):
      - GET /v1/models
      - GET /v1/audio/voices?model=<id>
      It refreshes every `refresh_interval` seconds.
    """
    _lock = threading.Lock()
    _models = None
    _voices_cache = {}
    _voice_discov = None  # per-server discovery instance
    _running = threading.Event()
    _stop_flag = False

    @classmethod
    def start(cls, server_url, api_key=None, timeout=10):
        with cls._lock:
            if cls._voice_discov is None:
                cls._voice_discov = cls(server_url, api_key, timeout)
                cls._running.set()

    @classmethod
    def stop(cls):
        cls._running.clear()

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._models = None
            cls._voices_cache.clear()

    @classmethod
    def get_models(cls, server_url, api_key=None):
        return cls._voice_discov.models() if cls._voice_discov else []

    @classmethod
    def get_voices(cls, server_url, model=None, api_key=None):
        return cls._voice_discov.voices(model) if cls._voice_discov else []


