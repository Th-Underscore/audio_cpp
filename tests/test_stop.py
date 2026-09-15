"""Hard-stop (interrupt) test: mock audio.cpp server + /voice/* routes.

Verifies, without a real GPU model:
  - client.stream_tts(cancel_event=...) aborts mid-stream WITHOUT raising
  - POST /voice/stop sets cancel_event, drains the chunk queue, and the
    worker skips every queued chunk (no further TTS requests are made)
  - the SSE relay emits a terminal `done` with file=null, stopped=true
  - /voice/stop 404s for an unknown chat id
"""
import base64
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from extensions.audio_cpp import client, relay


class MockAudioCpp(BaseHTTPRequestHandler):
    """Streams `n` PCM deltas per /v1/audio/speech request. Counts requests."""

    n_deltas = 20
    gap = 0.05
    lock = threading.Lock()
    requests = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._json({"ok": True})

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        with self.lock:
            MockAudioCpp.requests += 1
            req = MockAudioCpp.requests
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(self.n_deltas):
            pcm = bytes([0x11, 0x11]) * 100
            ev = {"type": "speech.audio.delta", "audio": base64.b64encode(pcm).decode()}
            self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
            self.wfile.flush()
            time.sleep(self.gap)
        self.wfile.write(("data: " + json.dumps({"type": "speech.audio.done"}) + "\n\n").encode())
        self.wfile.flush()


def start_mock(port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), MockAudioCpp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def start_routes(app, port):
    import uvicorn
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/voice/health" % port, timeout=1)
            return server
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("routes server did not come up")


def test_client_cancel():
    """cancel_event set mid-stream: generator returns cleanly, no raise."""
    srv = start_mock(5301)
    base = "http://127.0.0.1:5301"
    cancel = threading.Event()
    n = 0
    try:
        for _ in client.stream_tts(base, "m", "hi", cancel_event=cancel):
            n += 1
            cancel.set()  # abort after the first delta
    except Exception as e:
        raise AssertionError("stream_tts raised on cancel: %r" % e)
    assert 1 <= n < MockAudioCpp.n_deltas, n
    print("client cancel ok: %d deltas before abort (of %d)" % (n, MockAudioCpp.n_deltas))


def test_interrupt_session():
    mock_port, relay_port = 5302, 5303
    srv = start_mock(mock_port)
    base = "http://127.0.0.1:%d" % mock_port

    from fastapi import FastAPI
    app = FastAPI()
    sessions = {}
    cfg = {
        "server_url": base, "model": "m", "voice": None,
        "sample_rate": 24000, "chunk_min_chars": 10, "chunk_max_chars": 40,
        "save_file": False,
    }
    app.include_router(relay.make_routes(sessions, lambda: cfg))
    start_routes(app, relay_port)
    time.sleep(0.3)

    # unknown chat id -> 404
    req = urllib.request.Request(
        "http://127.0.0.1:%d/voice/stop?chat=nope" % relay_port, method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
        raise AssertionError("expected 404 for unknown chat")
    except urllib.error.HTTPError as e:
        assert e.code == 404, e.code
    print("/voice/stop 404-for-unknown ok")

    sess = relay.VoiceSession("stop1", cfg)
    sessions["stop1"] = sess
    long_text = ("The quick brown fox jumps over the lazy dog. " * 6).strip()
    # ~180 chars -> several chunks of max 40 chars at the chunker.
    sess.feed(long_text)

    # attach to the SSE relay (browser path)
    resp = urllib.request.urlopen(
        "http://127.0.0.1:%d/voice/stream?chat=stop1" % relay_port, timeout=20)
    time.sleep(0.8)  # let the first chunk synthesize
    n_before = MockAudioCpp.requests
    assert n_before >= 1, n_before

    # HARD stop
    req = urllib.request.Request(
        "http://127.0.0.1:%d/voice/stop?chat=stop1" % relay_port, method="POST")
    j = json.loads(urllib.request.urlopen(req, timeout=5).read())
    assert j.get("ok") is True, j

    sess.done_event.wait(timeout=5)
    assert sess.done_event.is_set()
    assert sess.cancel_event.is_set()

    # drain the SSE until the terminal done
    sse = b""
    while True:
        chunk = resp.read(4096)
        if not chunk:
            break
        sse += chunk
        if b"[DONE]" in sse:
            break
    text = sse.decode("utf-8", "replace")
    assert '"stopped": true' in text or '"stopped":true' in text, text
    assert '"file": null' in text or '"file":null' in text, text
    assert "[DONE]" in text, text
    n_audio = text.count("speech.audio.delta")
    assert n_audio >= 1, n_audio

    # the worker must have skipped all queued chunks: no TTS request after
    # the in-flight one.
    time.sleep(0.5)
    n_after = MockAudioCpp.requests
    assert n_after == n_before, (n_before, n_after)
    print("interrupt ok: %d deltas relayed, TTS requests %d -> %d (no new synthesis)"
          % (n_audio, n_before, n_after))
    print("ALL STOP TESTS PASS")


if __name__ == "__main__":
    test_client_cancel()
    test_interrupt_session()