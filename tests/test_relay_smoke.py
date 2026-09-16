"""Smoke test: mock audio.cpp server + Gradio-mounted /voice/* routes, e2e.

Verifies, without a real GPU model:
  - client.get_models / get_voices / stream_tts (SSE delta parsing)
  - make_routes endpoints served on a FastAPI app (same mount the Gradio
    app gets): /voice/health, /voice/active, /voice/stream, /voice/file
  - VoiceSession worker: feed -> synthesize -> out_q audio events -> done
  - save_opus (ffmpeg libopus) on real PCM
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
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json({"data": [{"id": "breeze-tts-2", "family": "breeze_tts"}]})
        elif self.path.startswith("/v1/audio/voices"):
            self._json({"data": [{"id": "femal-001"}]})
        else:
            self._json({"ok": True})

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(ln) or b"{}")
        # emit two pcm deltas: 200 samples of 0x1111, then 100 of 0x2222
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i, n in enumerate((200, 100)):
            pcm = bytes([0x11, 0x11]) * n if i == 0 else bytes([0x22, 0x22]) * n
            ev = {"type": "speech.audio.delta", "audio": base64.b64encode(pcm).decode()}
            self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
            self.wfile.flush()
        self.wfile.write(("data: " + json.dumps({"type": "speech.audio.done"}) + "\n\n").encode())
        self.wfile.flush()


def start_mock(port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), MockAudioCpp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def start_routes(app, port):
    """Serve a FastAPI app the same way Gradio serves its own (uvicorn)."""
    import uvicorn
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/voice/health" % port,
                                    timeout=1)
            return server
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("routes server did not come up")


def main():
    # --- client tests -------------------------------------------------------
    mock_port = 5099
    relay_port = 5199
    srv = start_mock(mock_port)
    base = "http://127.0.0.1:%d" % mock_port

    models = client.get_models(base)
    assert models == [("breeze-tts-2", "breeze_tts")], models
    voices = client.get_voices(base, "breeze-tts-2")
    assert voices == ["femal-001"], voices

    total = bytearray()
    for pcm in client.stream_tts(base, "breeze-tts-2", "Hello world.",
                                  options={"temperature": 0.9}, voice="femal-001"):
        total.extend(pcm)
    # 200*2 + 100*2 = 600 bytes
    assert len(total) == 600, len(total)
    print("client ok: models=%s voices=%s pcm=%dB" % (models, voices, len(total)))

    # --- routes + session test ----------------------------------------------
    # Mount the router on a bare FastAPI app, exactly as the deferred mount
    # in script.py does on the Gradio app.
    from fastapi import FastAPI
    app = FastAPI()
    sessions = {}
    cfg = {
        "server_url": base, "model": "breeze-tts-2", "voice": "femal-001",
        "sample_rate": 24000, "chunk_min_chars": 10, "chunk_max_chars": 40,
        "save_file": True,
    }
    app.include_router(relay.make_routes(sessions, lambda: cfg))
    rsv = start_routes(app, relay_port)
    time.sleep(0.3)

    h = urllib.request.urlopen("http://127.0.0.1:%d/voice/health" % relay_port).read()
    assert json.loads(h)["ok"] is True
    print("routes /health ok")

    h = urllib.request.urlopen("http://127.0.0.1:%d/voice/active" % relay_port).read()
    assert json.loads(h)["active"] is None
    print("routes /active (none) ok")

    # register a session, feed text, drain
    sess = relay.VoiceSession("smoke1", cfg)
    sessions["smoke1"] = sess
    h = urllib.request.urlopen("http://127.0.0.1:%d/voice/active" % relay_port).read()
    assert json.loads(h)["active"] == "smoke1", h
    print("routes /active (smoke1) ok")

    # attach to the live SSE relay over HTTP (browser path)
    resp = urllib.request.urlopen(
        "http://127.0.0.1:%d/voice/stream?chat=smoke1" % relay_port, timeout=15)
    sse = b""
    sess.feed("Hello world. How are you?")
    time.sleep(0.2)
    sess.finish()
    while True:
        chunk = resp.read(4096)
        if not chunk:
            break
        sse += chunk
        if b"[DONE]" in sse:
            break
    text = sse.decode("utf-8", "replace")
    assert "speech.audio.delta" in text, text
    assert '"type": "done"' in text or '"type":"done"' in text, text
    assert "[DONE]" in text, text
    print("routes /stream ok: %dB sse, %d deltas" %
          (len(sse), text.count("speech.audio.delta")))

    sess.done_event.wait(timeout=5)
    assert sess.done_event.is_set()

    # NOTE: the SSE endpoint above already consumed out_q (worker produces,
    # SSE reads). out_q is now empty — that double-observation is expected.
    events = []
    while True:
        try:
            events.append(sess.out_q.get_nowait())
        except Exception:
            break
    assert events == [], events
    n_audio = text.count("speech.audio.delta")
    assert n_audio >= 2, n_audio
    print("session pipeline ok: %d audio deltas relayed, stream ended with done" % n_audio)

    # verify an opus file was written and is served by /voice/file
    oggs = [f for f in os.listdir(relay.OUT_DIR) if f.endswith(".opus")] if os.path.isdir(relay.OUT_DIR) else []
    assert oggs, "no opus produced"
    size = os.path.getsize(os.path.join(relay.OUT_DIR, oggs[0]))
    assert size > 0
    file_url = "http://127.0.0.1:%d/voice/file?path=%s" % (relay_port, oggs[0])
    served = urllib.request.urlopen(file_url, timeout=10).read()
    assert len(served) == size, (len(served), size)
    print("opus ok: %s (%d bytes), /voice/file serves it" % (oggs[0], size))

    print("ALL RELAY SMOKE TESTS PASS")


if __name__ == "__main__":
    main()