"""Smoke test: mock audio.cpp server + Gradio-mounted /voice/* routes, e2e.

Verifies, without a real GPU model:
  - client.get_models / get_voices / stream_tts (SSE delta parsing)
  - make_routes endpoints served on a FastAPI app (same mount the Gradio
    app gets): /voice/health, /voice/active, /voice/stream, /voice/file
  - VoiceSession worker: feed -> synthesize -> out_q audio events -> done
  - transport flows: pcm (force-pcm baseline) and opus (auto -> libopus
    pipe; live packets in SSE, file demuxed back to the same packets)
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

# set by main() before any session is created
CFG = {}


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
        # emit two pcm deltas: 2400 samples of 0x1111, then 2400 of 0x2222.
# 2400 samples @24kHz mono = 4800B = 5 x 20ms frames (960B) each, so the
# opus path completes real frames (a sub-frame feed yields no packet).
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i, n in enumerate((2400, 2400)):
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


def run_stream_chat(sessions, port, chat_id, on_file=None):
    """Feed a session, drain its SSE relay over HTTP, return the raw text."""
    sess = relay.VoiceSession(chat_id, CFG, on_file=on_file)
    sessions[chat_id] = sess
    h = urllib.request.urlopen("http://127.0.0.1:%d/voice/active" % port).read()
    assert json.loads(h)["active"] == chat_id, h
    resp = urllib.request.urlopen(
        "http://127.0.0.1:%d/voice/stream?chat=%s" % (port, chat_id), timeout=15)
    sse = b""
    sess.feed("Hello world. How are you?")
    time.sleep(0.2)
    sess.finish()
    deadline = time.time() + 20
    while b"[DONE]" not in sse:
        chunk = resp.read(4096)
        if not chunk:
            break
        sse += chunk
        if time.time() > deadline:
            break
    sess.done_event.wait(timeout=5)
    assert sess.done_event.is_set()
    assert b"[DONE]" in sse, "stream did not end with [DONE]: %r" % sse[-200:]
    # out_q fully consumed by the SSE reader
    events = []
    while True:
        try:
            events.append(sess.out_q.get_nowait())
        except Exception:
            break
    assert events == [], events
    return sse.decode("utf-8", "replace")


def run_opus_flow(sessions, port):
    print("\n--- opus transport flow ---")
    # Regression: opus mode must fire on_file when the pipe's finish() writes
    # the .opus — the reaper records the registry entry from THAT callback
    # (unlike PCM, where save_opus fires it). Without it, the file is on disk
    # but never recorded and the client polls the registry for nothing (the
    # real-world "file not observed in registry after 6s").
    on_file_calls = []
    # auto is now gated on the capability assertion; re-post it so this
    # flow does not depend on execution order against the caps flow above.
    req = urllib.request.Request(
        "http://127.0.0.1:%d/voice/caps" % port,
        data=b'{"opus": true}', method="POST",
        headers={"Content-Type": "application/json"})
    assert json.loads(urllib.request.urlopen(req, timeout=5).read())["ok"]
    text = run_stream_chat(sessions, port, "smoke-opus",
                           on_file=on_file_calls.append)
    assert "speech.audio.start" in text, text
    # start is the FIRST data event and carries the transport
    ev = json.loads(text.split("data: ")[1])
    assert ev.get("transport") == "opus", ev
    n_opus = text.count("speech.audio.opus")
    assert n_opus >= 1, (n_opus, text)
    assert '"type": "done"' in text or '"type":"done"' in text, text
    assert "[DONE]" in text, text
    print("routes /stream (opus) ok: %dB sse, %d opus packets, start transport=opus"
          % (len(text), n_opus))
    # done event carries the file (first-delta commit)
    assert '"file": "' in text, "done event missing file"
    oggs = sorted([f for f in os.listdir(relay.OUT_DIR) if f.endswith(".opus")])
    assert oggs, "no opus produced"
    size = os.path.getsize(os.path.join(relay.OUT_DIR, oggs[-1]))
    assert size > 0
    file_url = "http://127.0.0.1:%d/voice/file?path=%s" % (port, oggs[-1])
    served = urllib.request.urlopen(file_url, timeout=10).read()
    assert len(served) == size, (len(served), size)
    print("opus file ok: %s (%d bytes), /voice/file serves it" % (oggs[-1], size))
    # the file must demux to >= as many packets as the stream delivered
    from extensions.audio_cpp import ogg_demux
    dem = ogg_demux.OggDemuxer()
    pkts = dem.feed(open(os.path.join(relay.OUT_DIR, oggs[-1]), "rb").read())
    assert len(pkts) >= n_opus, (len(pkts), n_opus)
    print("opus file demuxes: %d packets (stream delivered %d)" % (len(pkts), n_opus))
    # on_file must have fired for the pipe-finished file — this is what
    # makes the reaper record the entry the client polls for.
    assert len(on_file_calls) >= 1, "on_file never fired in opus mode"
    # (oggs[-1] above is the newest file in OUT_DIR, which may belong to the
    # caps-gate flow that ran earlier — so assert on_file fired for a file
    # named for THIS session, not the newest file.)
    assert any(c and os.path.basename(c).startswith("smoke-opus_")
               and c.endswith(".opus") for c in on_file_calls), on_file_calls
    print("on_file fired for opus file: %s" % on_file_calls[-1])


def _libopus_present():
    try:
        from extensions.audio_cpp import opus_pipe
        opus_pipe._get_lib()
        return True
    except Exception:
        return False


def run_pcm_flow(sessions, port):
    print("\n--- pcm transport flow (force-pcm) ---")
    CFG["live_transport"] = "force-pcm"
    text = run_stream_chat(sessions, port, "smoke-pcm")
    ev = json.loads(text.split("data: ")[1])
    assert ev.get("transport") == "pcm", ev
    n_audio = text.count("speech.audio.delta")
    assert n_audio >= 2, n_audio
    assert '"type": "done"' in text, text
    print("routes /stream (pcm) ok: %d pcm deltas relayed" % n_audio)
    CFG.pop("live_transport", None)


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
    # 2400*2 + 2400*2 = 9600 bytes
    assert len(total) == 9600, len(total)
    print("client ok: models=%s voices=%s pcm=%dB" % (models, voices, len(total)))

    # --- routes + session test ----------------------------------------------
    # Mount the router on a bare FastAPI app, exactly as the deferred mount
    # in script.py does on the Gradio app.
    from fastapi import FastAPI
    app = FastAPI()
    sessions = {}
    CFG.update({
        "server_url": base, "model": "breeze-tts-2", "voice": "femal-001",
        "sample_rate": 24000, "chunk_min_chars": 10, "chunk_max_chars": 40,
        "save_file": True,
    })
    app.include_router(relay.make_routes(sessions, lambda: dict(CFG)))
    rsv = start_routes(app, relay_port)
    time.sleep(0.3)

    h = urllib.request.urlopen("http://127.0.0.1:%d/voice/health" % relay_port).read()
    assert json.loads(h)["ok"] is True
    print("routes /health ok")

    h = urllib.request.urlopen("http://127.0.0.1:%d/voice/active" % relay_port).read()
    assert json.loads(h)["active"] is None
    print("routes /active (none) ok")

    # --- caps gate: auto transport -------------------------------------------
    # auto is server-side gated on the client's opus capability assertion
    # (player.js probes WebCodecs and POSTs /voice/caps). A fresh page has
    # not posted caps, so the FIRST auto session must resolve pcm; only
    # after the caps assertion do NEW sessions resolve opus (libopus present).
    assert relay._client_opus_capable is False, "caps flag not fresh at test start"
    text = run_stream_chat(sessions, relay_port, "smoke1")
    assert "speech.audio.start" in text, text
    assert '"type": "done"' in text, text
    assert "[DONE]" in text, text
    ev = json.loads(text.split("data: ")[1])
    assert ev.get("transport") == "pcm", ev
    assert "speech.audio.delta" in text, text
    assert "speech.audio.opus" not in text
    if _libopus_present():
        print("caps gate ok: auto + libopus, w/o /voice/caps -> pcm (%dB sse)"
              % len(text))
    else:
        print("caps gate ok (libopus absent): auto -> pcm (%dB sse)" % len(text))

    # POST /voice/caps the way player.js does after a successful probe
    req = urllib.request.Request(
        "http://127.0.0.1:%d/voice/caps" % relay_port,
        data=b'{"opus": true}', method="POST",
        headers={"Content-Type": "application/json"})
    caps = json.loads(urllib.request.urlopen(req, timeout=5).read())
    assert caps == {"ok": True, "opus": True}, caps
    assert relay._client_opus_capable is True
    print("routes /voice/caps ok:", caps)

    if _libopus_present():
        text = run_stream_chat(sessions, relay_port, "smoke1b")
        assert "speech.audio.start" in text, text
        assert '"type": "done"' in text, text
        assert "[DONE]" in text, text
        ev = json.loads(text.split("data: ")[1])
        assert ev.get("transport") == "opus", ev
        assert "speech.audio.opus" in text, text
        print("caps gate ok: auto + libopus + /voice/caps -> opus (%dB sse, %d opus pkts)"
              % (len(text), text.count("speech.audio.opus")))
    else:
        print("caps gate (libopus absent): auto stays pcm; opus phase skipped")

    # verify an opus file was written and is served by /voice/file
    oggs = [f for f in os.listdir(relay.OUT_DIR) if f.endswith(".opus")] \
        if os.path.isdir(relay.OUT_DIR) else []
    assert oggs, "no opus produced"
    size = os.path.getsize(os.path.join(relay.OUT_DIR, oggs[0]))
    assert size > 0
    file_url = "http://127.0.0.1:%d/voice/file?path=%s" % (relay_port, oggs[0])
    served = urllib.request.urlopen(file_url, timeout=10).read()
    assert len(served) == size, (len(served), size)
    print("opus ok: %s (%d bytes), /voice/file serves it" % (oggs[0], size))

    # --- opus transport flow --------------------------------------------------
    run_opus_flow(sessions, relay_port)

    # --- forced pcm flow ------------------------------------------------------
    run_pcm_flow(sessions, relay_port)

    print("\nALL RELAY SMOKE TESTS PASS")


if __name__ == "__main__":
    main()