// audio.cpp TTS player.
//
// Polls /voice/active (mounted on the SAME Gradio app, same origin) for the
// in-flight session, then attaches to that session's SSE stream and plays
// streamed PCM16 through a Web Audio playhead (playhead scheduling from
// tts_models/stream_client.py). When the session reports `done` with a
// finished .ogg, drops an <audio controls> player into the newest message
// for replay.
//
// No separate relay port: the /voice/* routes live on the Gradio app.

(function () {
    // Same origin as the chat page — the routes are mounted on this app.
    const RELAY_HOST = window.AUDIOCPP_RELAY_HOST || window.location.origin;
    const SAMPLE_RATE = window.AUDIOCPP_SAMPLE_RATE || 24000;
    const POLL_MS = 250;

    // Baked in by the server from the extension config (see script.custom_js).
    window.AUDIOCPP_ENABLED = window.AUDIOCPP_ENABLED || false;
    console.log("[audio_cpp] player.js loaded; enabled=%s relay=%s sr=%s",
                window.AUDIOCPP_ENABLED, RELAY_HOST, SAMPLE_RATE);

    let ctx = null;
    function audioCtx() {
        if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
        return ctx;
    }

    function b64ToBytes(b64) {
        const bin = atob(b64);
        const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out;
    }

    // Decode little-endian int16 mono PCM -> Float32 samples.
    function pcm16ToFloat(bytes) {
        const n = bytes.length >> 1;
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = view.getInt16(i * 2, true) / 32768.0;
        return out;
    }

    // Schedule one PCM chunk on the shared playhead; returns the new playhead.
    function playChunk(bytes, playhead) {
        if (bytes.length < 2) return playhead;
        const ac = audioCtx();
        if (ac.state === "suspended") ac.resume();
        const samples = pcm16ToFloat(bytes);
        const buf = ac.createBuffer(1, samples.length, SAMPLE_RATE);
        buf.getChannelData(0).set(samples);
        const src = ac.createBufferSource();
        src.buffer = buf;
        src.connect(ac.destination);
        const t = Math.max(playhead, ac.currentTime + 0.03);
        src.start(t);
        return t + samples.length / SAMPLE_RATE;
    }

    // If the live stream closes, allow re-attach: if /voice/active still
    // reports this session as in-flight, the stream died prematurely (e.g.
    // a stale relay) and the next poll should subscribe again.
    let reattach = 0;
    async function allowReattach(chatId) {
        try {
            const r = await fetch(RELAY_HOST + "/voice/active");
            if (!r.ok) return;
            const j = await r.json();
            if (j.ok && j.active === chatId) {
                reattach++;
                console.log("[audio_cpp] player: stream closed but session still ACTIVE — will re-attach (attempt %d)", reattach);
                current = null;
                if (reattach >= 3) {
                    console.warn("[audio_cpp] player: 3 re-attach attempts for %s; giving up", chatId);
                }
                return;
            }
        } catch (e) { /* route gone; nothing to re-attach to */ }
        if (current && current.chatId === chatId) current = null;
    }

    // Subscribe to one session's live audio.
    function streamFor(chatId, onDone) {
        const url = RELAY_HOST + "/voice/stream?chat=" + encodeURIComponent(chatId);
        console.log("[audio_cpp] player: opening stream (attach #%d)", chatId, reattach + 1);
        const es = new EventSource(url);
        let playhead = null;
        let nDeltas = 0;
        let closed = false;
        es.onopen = () => console.log("[audio_cpp] player: stream OPEN for", chatId);
        es.onerror = () => {
            if (closed) return;
            closed = true;
            console.warn("[audio_cpp] player: stream error/close, deltas=%d", nDeltas);
            allowReattach(chatId);
        };
        es.onmessage = (e) => {
            if (e.data === "[DONE]") {
                console.log("[audio_cpp] player: [DONE] after %d deltas", nDeltas);
                closed = true;
                es.close();
                allowReattach(chatId);
                return;
            }
            let ev;
            try { ev = JSON.parse(e.data); } catch (err) {
                console.warn("[audio_cpp] player: bad SSE payload:", e.data.slice(0, 200));
                return;
            }
            if (ev.type === "speech.audio.delta" && ev.audio) {
                const bytes = b64ToBytes(ev.audio);
                nDeltas++;
                if (nDeltas === 1 || nDeltas % 20 === 0) {
                    console.log("[audio_cpp] player: %d deltas, %dB, playhead=%s",
                                 nDeltas, bytes.length,
                                 playhead ? playhead.toFixed(2) + "s" : "null");
                }
                playhead = playChunk(bytes, playhead);
            } else if (ev.type === "done") {
                closed = true;
                es.close();
                if (onDone) onDone(ev.file || null);
            } else if (ev.type === "error") {
                console.warn("[audio_cpp] relay error:", ev.error);
                closed = true;
                es.close();
                allowReattach(chatId);
            }
        };
        return es;
    }

    function insertReplayer(fileUrl) {
        if (!fileUrl) return;
        // Resolve same-origin relative file URLs against the app origin.
        if (fileUrl.startsWith("/")) fileUrl = window.location.origin + fileUrl;
        // Newest completed message in the chat history.
        const msgs = document.querySelectorAll("#right-side .message.response, .message.response");
        const target = msgs.length ? msgs[msgs.length - 1] : document.querySelector("#right-side");
        if (!target) return;
        const holder = target.querySelector(".msg") || target;
        const a = document.createElement("audio");
        a.controls = true;
        a.className = "audio-cpp-player";
        a.src = fileUrl;
        holder.appendChild(a);
    }

    let current = null; // {chatId, seen: Set}

    let pollFailures = 0;

    async function poll() {
        // Not gated on the baked-in flag: it is only evaluated at page load,
        // and the user can toggle Enabled at runtime. The server only ever
        // advertises an active session when voicing is on, so an idle poll
        // costs one cheap local GET.
        try {
            const r = await fetch(RELAY_HOST + "/voice/active");
            if (!r.ok) throw new Error("HTTP " + r.status);
            pollFailures = 0;
            const j = await r.json();
            const active = j.ok ? j.active : null;
            if (!active) return;
            if (current && current.chatId === active) return;
            console.log("[audio_cpp] player: new active session", active);
            current = { chatId: active };
            streamFor(active, (fileUrl) => {
                console.log("[audio_cpp] player: session done, file=%s", fileUrl);
                insertReplayer(fileUrl);
            });
        } catch (e) {
            pollFailures++;
            if (pollFailures === 1 || pollFailures % 40 === 0) {
                console.warn("[audio_cpp] player: poll failed x%d (%s)",
                              pollFailures, e.message);
            }
            if (pollFailures >= 80) {
                console.warn("[audio_cpp] player: routes unreachable, stopping poll");
                window.clearInterval(timer);
            }
        }
    }

    window.AudioCppPlayer = { RELAY_HOST, SAMPLE_RATE, insertReplayer };
    // Expose a way for the extension to tell the page voicing is enabled.
    const timer = setInterval(poll, POLL_MS);
})();