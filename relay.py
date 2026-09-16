"""Per-generation voice sessions + Gradio-mounted SSE routes.

Topology (browser never touches audio.cpp):

    browser  --SSE-->  Gradio app /voice/*  --POST/SSE-->  audio.cpp
                           ^
                           | queue.Queue (audio/done events)
                           |
                      VoiceSession (per bot reply)
                           ^
                           | feed() from the custom_generate_reply wrapper
                           |
                      TextChunker + preprocessor

Routes are mounted directly on the Gradio FastAPI app (no separate port).
`make_routes(sessions, cfg_ref)` builds an APIRouter; the caller includes it
on `blocks.app` once the app object exists (see script.py deferred mount).

Endpoints:

    GET /voice/health
    GET /voice/active
    GET /voice/models
    GET /voice/voices?model=
    GET  /voice/stream?chat=<id>     SSE relay of the live session
    GET  /voice/file?path=<rel>      finished opus files (outputs/ dir)
    POST /voice/stop?chat=<id>       HARD stop: abort in-flight TTS now
"""

import json
import os
import queue
import subprocess
import threading
import time
import traceback
import urllib.parse
from typing import Callable, Dict

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from . import client
from . import chunker as chunker_mod
from . import preprocessor as preproc_mod
from . import specs

# Sentinel pushed on the input queue to tell the worker to finish.
_END = None

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Shared debug sink: set by the extension in setup() to shared.logger.
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


# --------------------------------------------------------------------------
# VoiceSession
# --------------------------------------------------------------------------
class VoiceSession:
    def __init__(self, chat_id, cfg, on_file=None, thinking_end_tag="",
                 cancel_event=None):
        self.chat_id = chat_id
        self.cfg = cfg                       # dict of current extension settings
        self.on_file = on_file               # optional cb(path) when file saved
        self._end_tag = thinking_end_tag     # thinking close tag ("" = none/auto)
        self._end_tag_seen = False
        self._suppress_logged = False
        self.chunker = chunker_mod.TextChunker(
            min_chars=cfg.get("chunk_min_chars", 120),
            max_chars=cfg.get("chunk_max_chars", 400),
            mode=cfg.get("chunk_mode", "sentence"),
            tag_pairs=cfg.get("chunk_tag_pairs", ""),
        )
        # TWO queues: the worker consumes `in_q` (chunks to synthesize) and
        # produces into `out_q` (audio/done events for the browser SSE).
        # Using one shared queue would let the worker and the SSE reader race.
        self.in_q = queue.Queue()
        self.out_q = queue.Queue()
        self._lock = threading.Lock()
        self._cancelled = False
        # Set by interrupt() (POST /voice/stop): client.stream_tts checks it
        # between deltas and closes the in-flight audio.cpp connection.
        # Owned by the session by default; a caller may pass its own Event.
        self.cancel_event = (cancel_event if cancel_event is not None
                             else threading.Event())
        self.ts = time.time()
        self.done_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._n_feed = 0
        self._n_chunks = 0
        fam = specs.detect_family(self.cfg.get("model", ""))
        _dbg("[session %s] created cfg={model=%r family=%s voice=%r sr=%s chunk=%s min=%s max=%s save=%s end_tag=%r}"
              % (self.chat_id, self.cfg.get("model"), fam or "UNKNOWN",
                self.cfg.get("voice"),
                self.cfg.get("sample_rate"), self.cfg.get("chunk_mode"),
                self.cfg.get("chunk_min_chars"), self.cfg.get("chunk_max_chars"),
                self.cfg.get("save_file"), self._end_tag))

    # -- fed by the LLM generation thread -----------------------------------
    def feed(self, cumulative_text):
        if self._end_tag:
            self._end_tag_seen = self._end_tag in cumulative_text
        clean = preproc_mod.clean(
            cumulative_text,
            strip_thinking=self.cfg.get("strip_thinking", True),
            strip_markdown=self.cfg.get("strip_markdown", True),
            strip_citations=self.cfg.get("strip_citations", True),
            thinking_end_tag=self._end_tag,
        )
        if (self._end_tag and not self._end_tag_seen
                and cumulative_text and not self._suppress_logged):
            self._suppress_logged = True
            _dbg("[session %s] suppressing streamed text: thinking close tag "
                 "not emitted yet (end-only thinking model)" % self.chat_id)
        self._n_feed += 1
        emitted = list(self.chunker.feed(clean))
        for c in emitted:
            self._n_chunks += 1
            self.in_q.put(c)
            _dbg("[session %s] chunk #%d queued: %r" % (self.chat_id, self._n_chunks, c))
        if self._n_feed <= 3 or emitted:
            _dbg("[session %s] feed #%d: raw_len=%d clean_len=%d chunks_out=%d"
                 % (self.chat_id, self._n_feed, len(cumulative_text), len(clean),
                    len(emitted)))

    def finish(self):
        """End of reply: flush the chunker tail, then close the pipeline."""
        with self._lock:
            tail = self.chunker.flush()
        for c in tail:
            self._n_chunks += 1
            self.in_q.put(c)
            _dbg("[session %s] flush chunk #%d queued: %r"
                 % (self.chat_id, self._n_chunks, c))
        if (self._end_tag and not self._end_tag_seen and self._n_chunks == 0):
            _dbg("[session %s] WARNING: thinking close tag never appeared; "
                 "entire reply treated as thinking, nothing voiced"
                 % self.chat_id)
        _dbg("[session %s] finish: feeds=%d chunks=%d pending_flush=%d end_tag_seen=%s"
             % (self.chat_id, self._n_feed, self._n_chunks, len(tail),
                self._end_tag_seen if self._end_tag else "-"))
        self.in_q.put(_END)

    def cancel(self):
        """DRAIN stop (the tap's GeneratorExit path: textgen Stop/Regenerate):
        flush the chunker tail, let the worker finish the in-flight TTS
        chunk, then close. Audio already relayed plays to the end."""
        with self._lock:
            self._cancelled = True
            tail = self.chunker.flush()
        for c in tail:
            self.in_q.put(c)
        self.in_q.put(_END)

    def interrupt(self):
        """HARD stop (browser stop button -> POST /voice/stop): abort NOW.

        Sets `cancel_event` (the in-flight client.stream_tts drops the
        audio.cpp connection at the next delta boundary), marks cancelled so
        every queued chunk is skipped, drains in_q, and emits a terminal
        `stopped` event on out_q so the browser SSE finalizes the playhead
        immediately. done_event is set here (not by the worker) so the
        caller's join is bounded; the worker's terminal `done` is a no-op
        because it only fires after the _END that interrupt() pushed, and
        _cancelled makes the worker skip straight to it.

        Idempotent: safe to call after the worker finished normally.
        """
        with self._lock:
            self._cancelled = True
        # Drop every queued chunk (don't synthesize the rest of the reply).
        while True:
            try:
                self.in_q.get_nowait()
            except queue.Empty:
                break
        self.cancel_event.set()
        self.in_q.put(_END)
        _dbg("[session %s] INTERRUPTED (hard stop): in-flight TTS abort, "
             "queued chunks dropped" % self.chat_id)
        # Terminal event: the SSE route converts ("stopped", None) into a
        # `done` with a null file (no opus file exists mid-reply) and breaks.
        self.out_q.put(("stopped", None))
        self.done_event.set()

    # -- worker thread -------------------------------------------------------
    def _run(self):
        try:
            self._run_inner()
        finally:
            self.done_event.set()

    def _run_inner(self):
        pcm_total = bytearray()
        sample_rate = int(self.cfg.get("sample_rate", 24000))
        file_path = None
        n_chunks = 0
        while True:
            text = self.in_q.get()
            if text is _END:
                _dbg("[session %s] worker: END sentinel after %d chunks"
                     % (self.chat_id, n_chunks))
                break
            if self._cancelled:
                _dbg("[session %s] worker: cancelled, skipping chunk %r"
                     % (self.chat_id, text[:60]))
                continue
            n_chunks += 1
            _dbg("[session %s] worker: synthesizing chunk #%d (%d chars): %r"
                 % (self.chat_id, n_chunks, len(text), text))
            try:
                for pcm in client.stream_tts(
                    self.cfg.get("server_url", "http://127.0.0.1:5023"),
                    self.cfg.get("model") or None,
                    text,
                    options=self._tts_options(),
                    voice=self.cfg.get("voice") or None,
                    seed=self._seed(),
                    timeout=int(self.cfg.get("request_timeout", 120)),
                    cancel_event=self.cancel_event,
                ):
                    pcm_total.extend(pcm)
                    self.out_q.put(("audio", b64(pcm)))
                _dbg("[session %s] worker: chunk #%d done, pcm_total=%dB"
                     % (self.chat_id, n_chunks, len(pcm_total)))
            except Exception as e:
                _dbg("[session %s] worker: TTS error on chunk #%d: %r"
                     % (self.chat_id, n_chunks, e))
                self.out_q.put(("error", str(e)))
                continue
        # persist the finished reply as opus
        if len(pcm_total) == 0:
            _dbg("[session %s] worker: finished with ZERO pcm — nothing to save"
                 % self.chat_id)
        if len(pcm_total) > 0 and self.cfg.get("save_file", True):
            try:
                os.makedirs(OUT_DIR, exist_ok=True)
                fname = "%s_%s.ogg" % (_safe(self.chat_id), int(time.time()))
                file_path = save_opus(pcm_total, sample_rate,
                                       os.path.join(OUT_DIR, fname))
                if file_path and self.on_file:
                    try:
                        self.on_file(file_path)
                    except Exception:
                        traceback.print_exc()
            except Exception:
                traceback.print_exc()
        _dbg("[session %s] worker: DONE chunks=%d pcm_total=%dB file=%s"
             % (self.chat_id, n_chunks, len(pcm_total), file_path))
        self.out_q.put(("done", file_path))
        self.done_event.set()

    def _tts_options(self):
        c = self.cfg
        opts = {
            "temperature": float(c.get("temperature", 0.9)),
            "depth_temperature": float(c.get("depth_temperature", 0.9)),
            "top_k": int(c.get("top_k", 0)),
            "top_p": float(c.get("top_p", 0.9)),
            "min_p": float(c.get("min_p", 0.0)),
            "guidance_scale": float(c.get("guidance_scale", 1.0)),
            "max_tokens": int(c.get("max_tokens", 1500)),
            "text_chunk_mode": c.get("text_chunk_mode", "default"),
            "text_chunk_size": int(c.get("text_chunk_size", 600)),
            "stream_frames_per_event": int(c.get("stream_frames_per_event", 1)),
            "stream_lookahead_margin": int(c.get("stream_lookahead_margin", 1)),
        }
        instr = (c.get("instruction") or "").strip()
        if instr:
            opts["instruction"] = instr
        ref = (c.get("reference_text") or "").strip()
        if ref:
            opts["reference_text"] = ref
        return opts

    def _seed(self):
        s = self.cfg.get("seed")
        if s in (None, "", "random", -1):
            return None
        try:
            return int(s)
        except (TypeError, ValueError):
            return None


def b64(data):
    import base64
    return base64.b64encode(data).decode("ascii")


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:60]


def save_opus(pcm, sample_rate, dest):
    """Encode PCM16-LE mono -> Ogg Opus via ffmpeg. Returns dest or None."""
    try:
        ffmpeg = os.environ.get("AUDIOCPP_FFMPEG", "ffmpeg")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error",
               "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
               "-c:a", "libopus", "-b:a", "64k", "-f", "ogg", dest]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        out, err = p.communicate(input=bytes(pcm), timeout=120)
        if p.returncode == 0 and os.path.exists(dest):
            _dbg("[save_opus] ok: %s (%d bytes pcm -> %d bytes ogg)"
                 % (dest, len(pcm), os.path.getsize(dest)))
            return dest
        _dbg("[save_opus] FAILED rc=%s err=%s" % (p.returncode,
                                                   (err or b"")[-400:].decode("utf-8", "replace")))
        return None
    except Exception as e:
        _dbg("[save_opus] exception: %r" % e)
        traceback.print_exc()
        return None


# --------------------------------------------------------------------------
# Voice library (stream_client.py pattern: <voice_dir>/*.wav + prompt_text)
# --------------------------------------------------------------------------
def list_voice_library(voice_dir):
    """Voices from <voice_dir>/*.wav joined with prompt_text transcripts.

    `prompt_text` is a plain-text file, one `name|transcript` per line.
    Returns (names, {name: transcript}).
    """
    import pathlib
    if not voice_dir:
        return [], {}
    vdir = pathlib.Path(voice_dir)
    if not vdir.is_dir():
        return [], {}
    texts = {}
    pf = vdir / "prompt_text"
    if pf.exists():
        for line in pf.read_text().splitlines():
            if "|" in line:
                name, text = line.split("|", 1)
                texts[name.strip()] = text.strip()
    names = [p.stem for p in sorted(vdir.glob("*.wav"))]
    # Per-voice transcript files: <name>.txt alongside <name>.wav
    for p in sorted(vdir.glob("*.txt")):
        if p.stem == "prompt_text":
            continue
        try:
            t = p.read_text().strip()
        except Exception:
            continue
        if t:
            texts[p.stem] = t
    return names, texts


# --------------------------------------------------------------------------
# Gradio route mounting
# --------------------------------------------------------------------------
def make_routes(sessions: Dict[str, VoiceSession],
                cfg_ref: Callable[[], dict]) -> APIRouter:
    """Build the /voice/* router. Include it on the Gradio app's FastAPI app."""
    router = APIRouter()

    def _cfg():
        try:
            return cfg_ref()
        except Exception:
            return {}

    def _server_url():
        return _cfg().get("server_url", "http://127.0.0.1:5023")

    @router.get("/voice/health")
    def health():
        return {"ok": True}

    @router.get("/voice/active")
    def active():
        # Newest session whose worker has not finished (in-flight audio).
        newest = None
        for sid, sess in sessions.items():
            if not sess.done_event.is_set():
                newest = sid
        return {"ok": True, "active": newest}

    @router.get("/voice/models")
    def models():
        try:
            return {"ok": True, "data": client.get_models(_server_url())}
        except Exception as e:
            return JSONResponse(status_code=502,
                                 content={"ok": False, "error": str(e)})

    @router.get("/voice/voices")
    def voices(model: str = Query(default=None)):
        try:
            server_voices = client.get_voices(_server_url(), model)
        except Exception as e:
            server_voices = []
            err = str(e)
        else:
            err = None
        lib_names, _ = list_voice_library(_cfg().get("voice_dir", ""))
        data, seen = [], set()
        for n in server_voices + lib_names:
            if n not in seen:
                seen.add(n)
                data.append(n)
        return {"ok": True, "data": data, "error": err}

    @router.get("/voice/voice-transcript")
    def voice_transcript(name: str = Query(default="")):
        _, texts = list_voice_library(_cfg().get("voice_dir", ""))
        return {"ok": True, "transcript": texts.get(name, "")}

    @router.get("/voice/stream")
    def stream(chat: str = Query(default=None)):
        if not chat or chat not in sessions:
            return JSONResponse(status_code=404,
                                 content={"ok": False, "error": "no session"})
        sess = sessions[chat]
        _dbg("[route /voice/stream] client subscribed chat=%s (session worker done=%s)"
             % (chat, sess.done_event.is_set()))

        def gen():
            n_ev = 0
            n_keepalive = 0
            yield b": connected\n\n"
            try:
                while True:
                    try:
                        kind, payload = sess.out_q.get(timeout=15)
                    except queue.Empty:
                        # Quiet window (e.g. slow first TTS call on a cold
                        # server): keep the SSE connection alive with a
                        # comment — NEVER treat an empty queue as EOF.
                        # (This used to be a blanket except -> [DONE] ->
                        # the browser's stream died before any audio.)
                        n_keepalive += 1
                        _dbg("[route /voice/stream] chat=%s keepalive #%d "
                             "(no event for 15s)" % (chat, n_keepalive))
                        yield b": keepalive\n\n"
                        continue
                    n_ev += 1
                    _dbg("[route /voice/stream] chat=%s ev #%d kind=%s payload_len=%d"
                         % (chat, n_ev, kind,
                            len(payload) if isinstance(payload, str) else -1))
                    if kind == "audio":
                        ev = {"type": "speech.audio.delta", "audio": payload}
                    elif kind == "error":
                        ev = {"type": "error", "error": payload}
                    elif kind == "done":
                        # Same-origin URL: the browser is already on this app.
                        file_url = None
                        if payload:
                            file_url = "/voice/file?path=%s" % urllib.parse.quote(
                                os.path.basename(payload))
                        ev = {"type": "done", "file": file_url}
                    elif kind == "stopped":
                        # Hard stop (interrupt()): terminal, no file (no opus
                        # file exists mid-reply). The browser finalizes the
                        # playhead and inserts its replayer.
                        ev = {"type": "done", "file": None, "stopped": True}
                    else:
                        continue
                    yield ("data: " + json.dumps(ev) + "\n\n").encode("utf-8")
                    if kind in ("done", "stopped"):
                        break
            except Exception:
                # generator abandoned (browser navigated away) — log it so a
                # future premature-close incident shows the real cause.
                _dbg("[route /voice/stream] chat=%s gen() EXC after %d evs, "
                     "%d keepalives:\n%s" % (chat, n_ev, n_keepalive,
                                              traceback.format_exc()))
                raise
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @router.post("/voice/stop")
    def stop(chat: str = Query(default=None)):
        """HARD stop: abort the in-flight TTS of the given session NOW
        (queued chunks dropped, in-flight request aborted)."""
        if not chat or chat not in sessions:
            return JSONResponse(status_code=404,
                                 content={"ok": False, "error": "no session"})
        sessions[chat].interrupt()
        return {"ok": True, "stopped": chat}

    @router.get("/voice/file")
    def serve_file(path: str = Query(default=None)):
        if not path:
            return JSONResponse(status_code=400,
                                 content={"ok": False, "error": "no path"})
        rel = os.path.normpath(path)
        if rel.startswith("..") or os.path.isabs(rel):
            return JSONResponse(status_code=400,
                                 content={"ok": False, "error": "bad path"})
        full = os.path.join(OUT_DIR, rel)
        if not os.path.exists(full) or not os.path.isfile(full):
            return JSONResponse(status_code=404,
                                 content={"ok": False, "error": "not found"})
        data = open(full, "rb").read()
        from fastapi.responses import Response
        return Response(content=data, media_type="audio/ogg")

    return router