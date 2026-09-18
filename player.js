// audio.cpp TTS — live-playhead streaming + per-message voice overlay.
//
// Live: polls /voice/active (same origin as the Gradio app), attaches to
// the session's SSE stream, and plays streamed PCM16 through a Web Audio
// playhead (playhead scheduling from tts_models/stream_client.py).
// Overlay: the server keeps a registry keyed assistant_<chatId>_<idx>_<v>
// -> {path, ts} in extensions/audio_cpp/audio_registry.json. On page load
// and after every session done, the client fetches /voice/registry and
// MIRRORS IT WHOLESALE into localStorage (server file is the single source
// of truth; delete hits the server directly).
// Inline players are derived at render time: an assistant row with
// data-index=N, displayed version k, in the currently-selected chat (read
// from the #past-chats Radio) shows one inline <audio> box iff the registry
// has assistant_<chatId>_<N>_<k>. Regeneration appends a new version, so
// each version of a row keeps its own file; each chat keeps its own.
//
// Consequences:
//   * refresh / chat-switch / version navigation: MutationObserver on #chat
//     re-derives players from the registry — idempotent, no stacking, no
//     loss. Rows whose box already matches the registry are left untouched
//     so our own writes never re-trigger the observer.
//   * regenerate: the server replaces the registry entry for the new row;
//     the old row keeps its own entry. No stale duplicates.
//
// No separate relay port: the /voice/* routes live on the Gradio app.

(function () {
    const RELAY_HOST = window.AUDIOCPP_RELAY_HOST || window.location.origin;
    const SAMPLE_RATE = window.AUDIOCPP_SAMPLE_RATE || 24000;
    const POLL_MS = 250;
    const LS_KEY = "audio_cpp.voices.v1";

    // Baked in by the server from the extension config (see script.custom_js).
    window.AUDIOCPP_ENABLED = window.AUDIOCPP_ENABLED || false;
    console.log("[audio_cpp] player.js loaded; enabled=%s relay=%s sr=%s",
                window.AUDIOCPP_ENABLED, RELAY_HOST, SAMPLE_RATE);

    // --- Web Audio playhead -------------------------------------------------
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
    function pcm16ToFloat(bytes) {
        const n = bytes.length >> 1;
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = view.getInt16(i * 2, true) / 32768.0;
        return out;
    }
    // Gap between streamed chunks. PADDING IS INJECTED SERVER-SIDE (relay
    // appends gap_ms of silence before every chunk after the first, and
    // into pcm_total so the saved file carries the same gaps) — the client
    // must not pad again, that would double the gap per chunk AND insert
    // silence between every event (deltas are sub-second slices, not
    // chunks).
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
    // Opus decode output (Float32Array from AudioData) into the same
    // playhead path. sampleRate/channels come from the decoded AudioData
    // (expected 24 kHz mono); mirrors playChunk's scheduling exactly.
    function playChunkF32(samples, playhead, sampleRate) {
        if (!samples || samples.length === 0) return playhead;
        const ac = audioCtx();
        if (ac.state === "suspended") ac.resume();
        const sr = sampleRate || SAMPLE_RATE;
        const buf = ac.createBuffer(1, samples.length, sr);
        buf.getChannelData(0).set(samples);
        const src = ac.createBufferSource();
        src.buffer = buf;
        src.connect(ac.destination);
        const t = Math.max(playhead, ac.currentTime + 0.03);
        src.start(t);
        return t + samples.length / sr;
    }

    // --- Opus capability probe (WebCodecs) -----------------------------------
    // Decides server-side auto resolution: only POST /voice/caps when a real
    // opus decode config is available. Runs once at load; failures are fatal
    // to the opus path for this page (streams fall back to pcm server-side).
    (async function probeOpusCapability() {
        if (typeof AudioDecoder === "undefined") {
            console.warn("[audio_cpp] AudioDecoder unavailable — opus incapable; " +
                         "not posting /voice/caps (auto -> pcm)");
            return;
        }
        let d = null;
        try {
            // AudioDecoder REQUIRES an init argument ({output, error}) —
            // new AudioDecoder() throws. Probe decoder discards output.
            d = new AudioDecoder({
                output() { /* probe: decoded data discarded */ },
                error: (e) =>
                    console.warn("[audio_cpp] opus capability probe error:", e)
            });
            // configure() is async — await it so a rejected configure also
            // counts as incapable (auto -> pcm).
            await d.configure({ codec: "opus", sampleRate: SAMPLE_RATE,
                                 numberOfChannels: 1 });
            d.close();
            console.log("[audio_cpp] opus capable (AudioDecoder configured @%d Hz mono)",
                        SAMPLE_RATE);
            fetch(RELAY_HOST + "/voice/caps", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ opus: true })
            }).then(r => r.json().then(j =>
                console.log("[audio_cpp] /voice/caps posted:", j)))
              .catch(e => console.warn("[audio_cpp] /voice/caps post failed:", e));
        } catch (e) {
            console.warn("[audio_cpp] opus decode config failed — incapable; " +
                         "auto will resolve pcm: " + e);
        }
    })();

    // --- Live-session SSE ---------------------------------------------------
    let reattach = 0;
    async function allowReattach(chatId) {
        try {
            const r = await fetch(RELAY_HOST + "/voice/active");
            if (!r.ok) return;
            const j = await r.json();
            if (j.ok && j.active === chatId) {
                reattach++;
                console.log("[audio_cpp] stream closed but session still ACTIVE — re-attach (attempt %d)", reattach);
                current = null;
                return;
            }
        } catch (e) { /* route gone */ }
        if (current && current.chatId === chatId) current = null;
    }

    function streamFor(chatId, onDone) {
        const url = RELAY_HOST + "/voice/stream?chat=" + encodeURIComponent(chatId);
        console.log("[audio_cpp] opening stream (attach #%d)", chatId, reattach + 1);
        const es = new EventSource(url);
        let playhead = null;
        let nDeltas = 0;
        let closed = false;
        // Opus transport (resolved server-side, announced by
        // speech.audio.start): per-stream decoder + 20 ms packet clock.
        let opusDecoder = null;
        let opusIdx = 0;
        let nOpus = 0;
        let opusDead = false;
        function closeOpus() {
            if (opusDecoder) {
                try { opusDecoder.flush(); } catch (e) { /* already closing */ }
                try { opusDecoder.close(); } catch (e) { /* already closed */ }
            }
            opusDecoder = null;
        }
        // Submit one opus packet to the decoder. decode() resolves with
        // UNDEFINED — decoded samples arrive asynchronously through the
        // decoder's output callback (see speech.audio.start setup above),
        // which feeds them to this stream's playhead via playChunkF32.
        // Awaiting the submit promise is the natural backpressure (the next
        // packet is only fed after this resolves).
        async function decoderDecode(decoder, chunk) {
            await decoder.decode(chunk);
        }
        es.onopen = () => console.log("[audio_cpp] stream OPEN for", chatId);
        es.onerror = () => {
            if (closed) return;
            closed = true;
            console.warn("[audio_cpp] stream error/close, deltas=%d opus=%d",
                         nDeltas, nOpus);
            closeOpus();
            hideStopBtn();
            allowReattach(chatId);
        };
        es.onmessage = (e) => {
            if (e.data === "[DONE]") {
                closed = true;
                es.close();
                closeOpus();
                hideStopBtn();
                allowReattach(chatId);
                return;
            }
            let ev;
            try { ev = JSON.parse(e.data); } catch (err) {
                console.warn("[audio_cpp] bad SSE payload:", e.data.slice(0, 200));
                return;
            }
            if (ev.type === "speech.audio.start") {
                if (ev.transport === "opus") {
                    try {
                        // AudioDecoder REQUIRES {output, error}. Decoded
                        // samples arrive ASYNCHRONOUSLY in output(); the
                        // feed side (decoderDecode) only tracks submit.
                        opusDecoder = new AudioDecoder({
                            // Decoded AudioData arrives here (async, in
                            // decode order, including closeOpus's flush
                            // tail) -> this stream's playhead, exactly as
                            // the pcm path.
                            output: (audio) => {
                                // AudioData has no getChannelData (that's
                                // AudioBuffer). Per MDN, copyTo(dest,
                                // {planeIndex, format}) extracts samples;
                                // an f32 TypedArray destination yields
                                // Float32 mono frames directly.
                                try {
                                    const buf = new Float32Array(
                                        audio.numberOfFrames);
                                    audio.copyTo(buf,
                                                  {planeIndex: 0,
                                                   format: "f32"});
                                    playhead = playChunkF32(
                                        buf, playhead, audio.sampleRate);
                                } catch (e) {
                                    console.warn("[audio_cpp] copyTo failed:",
                                                 e);
                                } finally {
                                    audio.close();
                                }

                            },
                            error: (e) => {
                                opusDead = true;
                                console.warn("[audio_cpp] opus decoder " +
                                             "error — stream is pcm-dead:", e);
                            }
                        });
                        opusDecoder.configure({ codec: "opus",
                                                 sampleRate: SAMPLE_RATE,
                                                 numberOfChannels: 1 });
                        console.log("[audio_cpp] opus stream start: decoder " +
                                    "configured @%d Hz mono", SAMPLE_RATE);
                    } catch (e) {
                        opusDecoder = null;
                        opusDead = true;
                        console.warn("[audio_cpp] stream is opus but decoder " +
                                     "unavailable/misconfigured — stream is " +
                                     "pcm-dead (opus events ignored): " + e);
                    }
                }
            } else if (ev.type === "speech.audio.opus" && ev.opus) {
                if (opusDead) return;
                if (!opusDecoder) {
                    console.warn("[audio_cpp] opus packet before start (or " +
                                 "decoder missing) — ignoring");
                    return;
                }
                const p = ev.opus.indexOf("|");
                const b64 = p >= 0 ? ev.opus.slice(0, p) : ev.opus;
                const bytes = b64ToBytes(b64);
                nOpus++;
                if (nOpus === 1 || nOpus % 20 === 0)
                    console.log("[audio_cpp] %d opus packets, %dB, playhead=%s",
                                 nOpus, bytes.length,
                                 playhead ? playhead.toFixed(2) + "s" : "null");
                // timestamp in microseconds; every packet is exactly 20 ms
                // at 24 kHz mono, so the packet index IS the clock.
                const chunk = new EncodedAudioChunk({
                    type: "key",
                    timestamp: opusIdx * 20000,
                    data: bytes
                });
                opusIdx++;
                decoderDecode(opusDecoder, chunk).catch(e => {
                    console.warn("[audio_cpp] opus decode error:", e);
                    opusDead = true;
                });
            } else if (ev.type === "speech.audio.delta" && ev.audio) {
                const bytes = b64ToBytes(ev.audio);
                nDeltas++;
                if (nDeltas === 1 || nDeltas % 20 === 0)
                    console.log("[audio_cpp] %d deltas, %dB, playhead=%s",
                                 nDeltas, bytes.length,
                                 playhead ? playhead.toFixed(2) + "s" : "null");
                playhead = playChunk(bytes, playhead);
            } else if (ev.type === "done") {
                closed = true;
                es.close();
                hideStopBtn();
                if (ev.stopped) {
                    current = null;
                    return;
                }
                onSessionDone(ev.file || null);
            } else if (ev.type === "error") {
                console.warn("[audio_cpp] relay error:", ev.error);
                closed = true;
                es.close();
                hideStopBtn();
                allowReattach(chatId);
            }
        };
        return es;
    }

    // --- Voice overlay: registry mirror + chips -----------------------------
    let clientVoices = loadLocalVoices();

    function loadLocalVoices() {
        try {
            const raw = localStorage.getItem(LS_KEY);
            const j = raw ? JSON.parse(raw) : {};
            return (j && typeof j === "object") ? j : {};
        } catch (e) {
            console.warn("[audio_cpp] localStorage load failed:", e);
            return {};
        }
    }
    function saveLocalVoices() {
        try { localStorage.setItem(LS_KEY, JSON.stringify(clientVoices)); }
        catch (e) { console.warn("[audio_cpp] localStorage save failed:", e); }
    }

    // Server registry is the source of truth; the client mirrors it
    // wholesale. fetchRegistry is a SINGLE-FLIGHT fetch: concurrent callers
    // share one in-flight request (no dropped requests), each has a 3s
    // AbortController timeout so a slow/hung request can never gate all
    // future syncs. The mirror is only updated when the response differs
    // from the current one, so the low-frequency poller below never causes
    // render churn.
    //
    // Two users of the mirror:
    // * a 400ms page-load poller keeps the mirror fresh across tabs (a
    //   delete in another tab shows up here within ~0.4s);
    // * onSessionDone polls until it positively observes the reaper's new
    //   record (the server writes it a moment AFTER the done event) —
    //   confirmation, not blind retries.
    //
    // mirrorFresh: true once a fetch has succeeded. renderOverlay may only
    // REMOVE a box when the mirror is fresh — otherwise a render racing
    // ahead of the first sync (page load / chat switch) sees a stale/empty
    // mirror and would delete freshly-synced players.
    let syncing = false;
    let mirrorFresh = false;
    async function fetchRegistry() {
        const ctrl = new AbortController();
        const to = setTimeout(() => ctrl.abort(), 3000);
        try {
            const r = await fetch(RELAY_HOST + "/voice/registry", { signal: ctrl.signal });
            if (!r.ok) throw new Error("HTTP " + r.status);
            return await r.json();
        } finally { clearTimeout(to); }
    }
    async function syncRegistry() {
        if (syncing) return; // concurrent caller is already fetching
        syncing = true;
        try {
            const j = await fetchRegistry();
            if (j && typeof j === "object" &&
                JSON.stringify(j) !== JSON.stringify(clientVoices)) {
                clientVoices = j;
                saveLocalVoices();
                mirrorFresh = true;
                console.log("[audio_cpp] registry synced:",
                            JSON.stringify(Object.keys(j)));
                queueRender();
            }
        } catch (e) {
            console.warn("[audio_cpp] registry fetch failed:", e);
        } finally { syncing = false; }
    }
    function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

    // Keep the mirror fresh across tabs: poll every 400ms while visible.
    // Cheap: one small JSON GET per 0.4s, and syncRegistry no-ops render
    // when the payload is unchanged.
    setInterval(() => {
        if (!document.hidden) syncRegistry();
    }, 400);

    // On session done: the reaper records the finished file a moment AFTER
    // the done event. Poll until we positively observe a registry entry
    // whose path is the finished file (the filename stem is unique) —
    // confirmation, not blind retries; max ~6s.
    async function onSessionDone(fileUrl) {
        const m = fileUrl && fileUrl.match(/path=([^&]+)/);
        const stem = m ? decodeURIComponent(m[1]).replace(/\.(ogg|opus)$/, "") : null;
        console.log("[audio_cpp] session done, file=%s — polling registry", fileUrl);
        for (let i = 0; i < 15; i++) {
            try {
                const j = await fetchRegistry();
                if (j && typeof j === "object") {
                    if (JSON.stringify(j) !== JSON.stringify(clientVoices)) {
                        clientVoices = j;
                        saveLocalVoices();
                        mirrorFresh = true;
                        console.log("[audio_cpp] registry synced:",
                                    JSON.stringify(Object.keys(j)));
                    }
                    queueRender();
                    if (!stem || Object.values(j).some(v => v.path &&
                            (v.path.endsWith(stem + ".ogg") ||
                             v.path.endsWith(stem + ".opus"))))
                        return; // observed the reaper's record
                }
            } catch (e) { /* server busy; keep polling */ }
            await sleep(400);
        }
        console.warn("[audio_cpp] session done: file not observed in registry after 6s");
    }

    // --- DOM helpers ---------------------------------------------------------
    function assistantMessages() {
        return Array.from(document.querySelectorAll("#chat .message"))
            .filter(el => el.querySelector(".circle-bot"));
    }
    function msgIdx(el) {
        const v = el && el.dataset && el.dataset.index;
        return (v !== undefined && v !== null && v !== "") ? Number(v) : null;
    }
    function msgVersion(el) {
        const vp = el && el.querySelector(".version-position");
        if (!vp) return 1;
        const m = (vp.textContent || "").trim().match(/^(\d+)/);
        return m ? Number(m[1]) : 1;
    }

    // Active chat's unique id, read from the #past-chats Radio. Gradio
    // renders each option's value as the input value; the selected option
    // is :checked. Empty when nothing is selectable (incognito / brand-new
    // unsaved chat) — the key then simply never matches, so no player.
    function currentChatId() {
        const radio = document.querySelector("#past-chats");
        if (!radio) return "";
        const sel = radio.querySelector("input:checked");
        return sel ? (sel.value || "") : "";
    }

    // Key for an assistant row in the current view:
    // assistant_<chatId>_<data-index>_<displayed version>.
    function rowKey(idx, version) {
        return "assistant_" + currentChatId() + "_" + idx + "_" + version;
    }

    // --- Inline voice player rendering ---------------------------------------
    // Each assistant row with a registry entry gets ONE inline <audio> box
    // under its .text: [voice vN] <audio controls> [delete]. Render is
    // idempotent: a row whose box already matches the registry entry is left
    // untouched, so the MutationObserver on #chat never sees our own writes
    // and the render is a no-op until the registry actually changes.
    function renderOverlay() {
        for (const el of assistantMessages()) {
            const idx = msgIdx(el);
            if (idx === null) continue;
            const v = msgVersion(el);
            const key = rowKey(idx, v);
            const entry = clientVoices[key];
            const box = el.querySelector(".audio-cpp-voice");
            if (box && entry && box.dataset.src === entry.path) continue; // up to date
            // No entry in the mirror: only remove an existing box when the
            // mirror is a confirmed server state (sync/delete completed).
            // While the mirror is stale we simply leave the box alone —
            // e.g. a render fired on chat-switch before the sync resolved.
            if (!entry && box) { if (mirrorFresh) box.remove(); continue; }
            if (!entry) continue;
            if (box) box.remove();
            const holder = el.querySelector(".text") || el;
            const div = document.createElement("div");
            div.className = "audio-cpp-voice";
            div.dataset.src = entry.path;
            const a = document.createElement("audio");
            a.controls = true;
            a.preload = "metadata";
            const abs = entry.path.startsWith("http")
                ? entry.path
                : RELAY_HOST + "/voice/file?path=" + encodeURIComponent(entry.path);
            a.src = abs;
            const del = document.createElement("button");
            del.className = "aac-del";
            del.textContent = "delete";
            del.title = "Delete this voice (server file entry)";
            del.addEventListener("click", async (e) => {
                e.stopPropagation();
                const resp = await fetch(RELAY_HOST + "/voice/registry?key=" +
                                          encodeURIComponent(key), { method: "DELETE" });
                const j = await resp.json().catch(() => ({}));
                console.log("[audio_cpp] delete voice:", key, j);
                if (j.ok && j.deleted) {
                    delete clientVoices[key];
                    saveLocalVoices();
                    queueRender();
                }
            });
            div.appendChild(a);
            div.appendChild(del);
            holder.appendChild(div);
        }
    }

    // Watch #chat for structural changes (new messages, version nav,
    // refresh re-render, chat switch). Debounce burst mutations.
    let renderQueued = false;
    function queueRender() {
        if (renderQueued) return;
        renderQueued = true;
        setTimeout(() => { renderQueued = false; renderOverlay(); }, 80);
    }
    const chat = document.querySelector("#chat");
    if (chat) {
        const mo = new MutationObserver(() => queueRender());
        mo.observe(chat, { subtree: true, childList: true, characterData: true,
                            attributes: true, attributeFilter: ["data-index"] });
    }
    // Cross-tab sync.
    window.addEventListener("storage", (e) => {
        if (e.key === LS_KEY) { clientVoices = loadLocalVoices(); queueRender(); }
    });

    // --- Stop button --------------------------------------------------------
    let stopBtn = null, current = null;
    function ensureStopBtn() {
        if (stopBtn) return stopBtn;
        stopBtn = document.createElement("button");
        stopBtn.id = "audio-cpp-stop";
        stopBtn.textContent = "■ Stop voice";
        stopBtn.className = "audio-cpp-stop-btn";
        stopBtn.style.cssText =
            "position:fixed;bottom:24px;right:24px;z-index:9999;" +
            "background:#c0392b;color:#fff;border:none;border-radius:8px;" +
            "padding:10px 18px;font-size:14px;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.35);";
        stopBtn.addEventListener("click", () => {
            if (!current || !current.chatId) return;
            console.log("[audio_cpp] stop:", current.chatId);
            fetch(RELAY_HOST + "/voice/stop?chat=" + encodeURIComponent(current.chatId),
                  { method: "POST" })
                .then(r => r.json())
                .then(j => console.log("[audio_cpp] stop:", j))
                .catch(e => console.warn("[audio_cpp] stop failed:", e));
        });
        document.body.appendChild(stopBtn);
        return stopBtn;
    }
    function showStopBtn(chatId) {
        ensureStopBtn();
        stopBtn.style.display = "";
        stopBtn.textContent = "■ Stop voice";
        stopBtn.dataset.chat = chatId;
    }
    function hideStopBtn() {
        if (stopBtn) stopBtn.style.display = "none";
    }

    // --- Poll loop ----------------------------------------------------------
    let pollFailures = 0;
    async function poll() {
        try {
            const r = await fetch(RELAY_HOST + "/voice/active");
            if (!r.ok) throw new Error("HTTP " + r.status);
            pollFailures = 0;
            const j = await r.json();
            const active = j.ok ? j.active : null;
            if (!active) {
                if (current) { current = null; hideStopBtn(); }
                return;
            }
            if (current && current.chatId === active) return;
            if (current && current.es) {
                try { current.es.close(); } catch (e) {}
                current = null;
            }
            console.log("[audio_cpp] new active session", active);
            const es = streamFor(active, onSessionDone);
            current = { chatId: active, es };
            showStopBtn(active);
        } catch (e) {
            pollFailures++;
            if (pollFailures === 1 || pollFailures % 40 === 0)
                console.warn("[audio_cpp] poll failed x%d (%s)", pollFailures, e.message);
            if (pollFailures >= 80) {
                console.warn("[audio_cpp] routes unreachable, stopping poll");
                window.clearInterval(timer);
            }
        }
    }

    // Initial render (covers page load with an existing chat) + initial
    // registry sync + start poll.
    renderOverlay();
    syncRegistry();
    window.AudioCppPlayer = { RELAY_HOST, SAMPLE_RATE, renderOverlay,
                               syncRegistry, voices: () => clientVoices };
    const timer = setInterval(poll, POLL_MS);
})();