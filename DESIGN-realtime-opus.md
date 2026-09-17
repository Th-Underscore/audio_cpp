# audio_cpp — realtime Opus transport (live voice)

Design for streaming **compressed Opus** to the browser during synthesis,
replacing raw PCM16 on the live path, while the saved `.opus` file falls out
of the *same* encoder for free.

Status: design (approved shape); implementation pending.

## Motivation

1. **Network bandwidth.** The live relay currently carries raw PCM16 mono at
   24 kHz = **384 kbps**. At 64 kbps Opus that is ~6× less. On loopback this is
   irrelevant; for textgen accessed over the internet it is a real cost, and a
   minimal UI should stay true to minimal transport.
2. **Pipeline simplification (emergent).** An Ogg/Opus byte stream is already
   a valid file. If the live encoder's stdout is tee'd to disk as it goes, the
   post-stream batch ffmpeg call in `save_opus` disappears entirely — one
   encoder per session does both jobs.

## Grounding (what was verified)

- **24 kHz is a native Opus rate** (RFC 6716 supports 8–48 kHz; ffmpeg's
  `libopus` lists 24000 in `libopus_sample_rates`). Ogg mapping declares it via
  the `up/channel` flags (super-wideband). **No resampling.** The earlier
  "Opus only accepts 48 kHz" claim applied only to the codec's internal 48 kHz
  clock, not the container/stream layer.
- **Raw-packet WebCodecs decode is the old path, not the new one.**
  W3C WebCodecs Opus registration: `AudioDecoder` accepts two bitstream
  formats — with `description` (OpusHead) set the packets are Ogg-encapsulated
  (needs `OggParser`, Chrome 131+); **without it, packets are raw RFC 6716
  Opus packets** and the config is simply
  `{codec:'opus', sampleRate:24000, numberOfChannels:1}`. `description` is
  only required in Chromium for >2 channels. → **No `OggParser` dependency;
  floor is Chrome 94+ (2021), Firefox 130+ (2024), Safari 16.4+ (partial;
  26+ full, ~68% macOS Safari in Jan 2026 sample data).**
- **History check:** Opus has been in browsers since ~2013 (WebRTC MTI) and
  `<audio src=*.ogg>` since ~2016, but WebRTC-era decode lives in the *native*
  media stack — JavaScript never sees the Opus bytes. WebCodecs (2021) is the
  first time JS can decode a raw Opus stream; that is the floor we rely on.
- `pyopus` is a CPython C-extension (not pyo3) → does not build on
  Python 3.12+. **Decision: subprocess ffmpeg**, as user-suggested. Its
  stdin/stdout pipe is simple and performant enough.

## Architecture

```
                       ffmpeg: -f s16le -ar 24000 -ch_layout mono -i -
                               -c:a libopus -b:a 64k -f ogg -
PCM delta ────stdin──────────► (one subprocess per voice session)
                              stdout = Ogg page stream
                                   │
                            tee thread
                          ┌────────┴─────────┐
                          ▼                   ▼
                append → <chat>_<ts>.opus   demux pages → raw Opus packets
                (valid file, free)              │
                                            ┌───┴───┐
                                            ▼       ▼
                                     mode=opus  mode=pcm
                                    1 SSE event per raw packet (b64)
                                    ORIGINAL pcm delta, untouched
                                    (encoder output ignored)
```

- The **file and the live stream differ by exactly the container**: the file
  is the real Ogg stream (what the overlay `<audio>` plays); the stream is the
  raw packets inside it. Both decode to identical audio.
- The **PCM mode is literally today's behavior** — same events, same player
  code — so "fallback" adds zero new client playback code, only a mode flag.
- `save_opus` (post-stream batch encode) is **deleted**; the file *is* the tee
  output. On interrupt the file is a valid prefix of the stream (may end
  mid-page; optionally truncate to the last complete page boundary).

### Server changes (relay.py / script.py)

1. **Encoder lifecycle:** spawn one ffmpeg subprocess per voice session at
   first PCM delta; stdin fed per delta; stdout drained by a reader thread;
   kill/terminate on session end, interrupt, and reply stop.
2. **Tee thread:** append stdout bytes to the `.opus` file (when `save_files`
   on) AND demux Ogg pages → forward per transport mode:
   - `opus`: one SSE `audio` event per raw packet (base64)
   - `pcm`: the original PCM delta (current behavior)
3. **Ogg demux** (~30 lines): 27-byte page header + lacing table; a logical
   packet ends where a lacing value < 255; first page's two header packets
   (`OpusHead`/`OpusTags`) are skipped for the stream.
4. **SSE event changes:**
   - `start` gains `transport: "opus"|"pcm"` (the resolved mode for this
     stream). For opus the client configures the decoder in **raw-packet
     format** (no `description`/OpusHead), with rate/channels from
     `start.sample_rate`/`channels` as today.
   - `audio` event is mode-dependent: opus = one 20 ms packet (b64); pcm =
     delta bytes (b64). `timestamp` semantics: opus packets are sequential
     20 ms (client derives `t = n * 0.020`).
   - `done` unchanged (`file` path of the tee'd `.opus`).
5. **Mode resolution:** request param `transport=auto|opus|pcm`
   (client sends its resolved mode); server defaults to `pcm` for unknown
   values. `auto` resolution happens client-side (see below); server treats
   `auto` as `opus` only if the client also asserts capability — simpler:
   client never sends `auto`.

### Client changes (player.js)

1. **Capability probe (once, at load):** `typeof AudioDecoder !== 'undefined'`
   AND `new AudioDecoder({output(){}, error(){}})` + `configure({codec:'opus',
   sampleRate:24000, numberOfChannels:1})` resolves → capable.
   (The constructor REQUIRES an AudioDecoderInit {output, error}; a bare
   `new AudioDecoder()` throws in every WebCodecs browser. A failed
   configure rejects → incapable.)
2. **Effective transport:** settings mode × capability:
   - `auto` (default): opus if capable, else pcm
   - `opus`: opus (trust the user; if the probe says incapable, warn and
     fall back to pcm for this stream)
   - `pcm`: always pcm
3. **Opus playback path:** on `audio` event,
   `new EncodedAudioChunk({format:'opus', data, timestamp})` →
   `AudioDecoder.decode(chunk)`. decode() enqueues and resolves with
   undefined; decoded samples arrive ASYNCHRONOUSLY in the decoder's
   `output` callback as AudioData. Per MDN, AudioData has no
   getChannelData — samples are extracted with
   `copyTo(Float32Array(numberOfFrames), {planeIndex: 0, format: 'f32'})`
   (mono f32) → existing playChunk playhead-scheduling path, untouched.
   - `AudioDecoder` created per stream (configure → decode… → flush/close);
     backpressure by awaiting each decode submit (decodeQueueSize bounds the
     queue).
4. **PCM playback path:** unchanged (raw PCM → `AudioBuffer` → play).
5. The rest of player.js (SSE fetch loop, registry sync, overlay, stop)
   untouched.

### Settings (script.py accordion, settings.json)

- `live_transport` — `auto` (default) | `opus` | `pcm`, a new "Live voice
  transport" dropdown. Persisted like all other keys. Mode changes affect
  only **new** replies (per-session resolution at stream start) — correct
  granularity.
- `opus_bitrate_kbps` (64) now governs the live encoder too (single value;
  keeps file and stream byte-identical in content).

### Safari / edge-case rationale

WebCodecs Opus is only *partial* on Safari (16.4 incomplete, 26 full, ~68%
of macOS Safari in Jan 2026 data). For Safari users the `pcm` mode is likely
the *primary* path until 26 is widespread — hence auto-detect + manual toggle
rather than auto alone. Non-WebCodecs browsers lose **live** audio in opus
mode only; saved `.opus` files keep playing everywhere via `<audio>` exactly
as today, so the overlay/registry remains fully functional in old browsers.

## Trade-offs accepted

- **~100–200 ms added first-audio latency** (ffmpeg buffers whole Ogg pages
  ≈ 100 ms of audio before writing) vs. raw-PCM zero-copy. Synthesis time per
  chunk dominates first-audio delay, so this is acceptable.
- **Browser floor for live opus:** Chrome 94 / Firefox 130 / Safari 16.4(p).
  Deliberate, covered by the toggle.
- **One long-lived subprocess per session** (was one batch call per reply).
  Process-leak risk is bounded by session-end/interrupt/timeout teardown;
  a watchdog on the reader thread's liveness terminates strays.

## Out of scope

- WebRTC-style congestion/loss handling, FEC, DTX — single-client loopback /
  direct connection, not a conference call.
- Client-side Ogg demux — server does it; client sees raw packets only.
- Multichannel — TTS is mono.