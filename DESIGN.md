# audio_cpp — streaming TTS extension for textgen

Design document. Target: realtime text-to-speech for streamed LLM replies via any
audio.cpp server (`http://<host>:<port>`), with paragraph/sentence chunking
mid-generation so audio starts ~1–2 sentences into a reply instead of after the
full reply is done.

Reference implementation studied:
`/mnt/c/.../tts_models/stream_client.py` (standalone, fully featured audio.cpp SSE client).

## Pipeline

```
Browser (player.js, WebAudio)
   │  fetch(SSE) same-origin (no extra port)
   ▼
Gradio app  /voice/* routes  (mounted on blocks.app; no separate server)
   │  GET /voice/stream?chat=<name>  →  SSE: audio delta b64 frames
   ▼
Per-conversation worker thread
   │  drains chunk queue, sequential TTS calls
   ▼
audio.cpp server  (user-run, external; extension's ONLY outbound TTS traffic)
   POST /v1/audio/speech  {response_format: pcm, stream_format: sse, ...}
```

- Client NEVER talks to audio.cpp directly. Only the Gradio app does.
- textgen's LLM stream is tapped via a **`custom_generate_reply` wrapper**
  (RESOLVED — see below). `output_modifier` does NOT fire per-yield
  (`apply_extensions('output')` is called only once at the END,
  text_generation.py:126-128), so it can't see mid-generation text. Instead
  the extension defines `custom_generate_reply(question, original_question,
  state, stopping_strings, is_chat=False)`; `_generate_reply` picks it up via
  `apply_extensions('custom_generate_reply')` (text_generation.py:49) and calls
  it at :96. The wrapper dispatches to the real base generator
  (`generate_reply_HF` / `generate_reply_custom` from modules.text_generation,
  chosen by model class exactly as `_generate_reply` does at :56-59) and
  wraps its yields: each accumulated string is re-yielded UNCHANGED (so the
  UI is byte-identical) while the *diff* since the last yield is fed to the
  per-conversation chunker. The chunker is O(new chars) and never blocks the
  UI. dayna_ss does NOT define `custom_generate_reply` (it defines
  `generate_chat_reply`, a different hook), so this tap is free.
- The `/voice/*` routes are mounted **directly on the Gradio FastAPI app** —
  NO separate relay port. In the custom gradio 4.37 build the app object
  (`blocks.app`) is created in `Blocks.__exit__` AND recreated in `queue()`
  AND again in `launch()` (gradio/blocks.py:2055/2114/2275 — each call
  replaces `self.app` with a fresh FastAPI; the SERVED app is the one
  `launch()` builds). Extension `setup()` runs before all three — so a
  deferred thread polls `shared.gradio['interface'].app` and re-mounts
  `include_router(relay.make_routes(...))` every time the app's object
  identity changes. (Mounting once on the first app and stopping was the
  404 bug: the router landed on the discarded pre-launch app.) Because the
  routes live on
  the app the browser is already connected to, the browser talks only to the
  Gradio origin (no CORS, no second port). `make_routes(sessions, cfg_ref)`
  builds a `fastapi.APIRouter`; the worker threads and `VoiceSession` are
  unchanged from the relay design.

## Extension hooks used

| Hook | Purpose |
|---|---|
| `setup()` | read settings, start deferred route-mount thread (mounts /voice/* on `blocks.app` once it exists) |
| `ui()` | accordion in the Extensions tab |
| `custom_js()` | return player.js inline |
| `custom_generate_reply(...)` | the stream tap: wraps base generator, feeds chunker, re-yields unchanged |
| `output_modifier(string, state)` | end-of-reply: persist the opus file ref / handle empty-reply case (no-op for streaming path) |

## Chunker (chunker.py)

Stateful per conversation. Feeding `output_modifier` during a stream yields
newly-*completed* units:

- **default mode (sentence):** split on `. ! ? …` and `:`, and on blank lines.
  Accumulate; emit when the buffer ≥ `min_chunk_chars` (default 120) at a
  split point, or immediately at `max_chunk_chars` (default 400) with a hard
  split (never mid-word; prefer last whitespace before the cap).
- **paragraph mode:** emit per blank-line-separated paragraph (same min/max
  coalescing/overflow rules).
- **custom modes (TBD):** chunker is a strategy object selected by
  `chunking_mode`; adding a mode = adding a class. The upstream
  `text_chunk_mode` is a separate passthrough field (audio.cpp-side
  chunking) and remains fully configurable.
- On the final non-stream pass (or `is_stream=False` tail), flush the
  remainder as the last chunk.
- New generation for the same chat resets state; a new chat key is derived
  from `chat.name` (+ counter) so parallel/queued replies don't cross-contaminate.

## Preprocessing (preprocessor.py)

Applied per chunk before TTS, each toggle independently configurable:

- strip thinking blocks (`<thinking>...</thinking>`, `...`, per
  modules/reasoning.py formats; the model here is non-thinking but kept safe)
- strip tool-call markers (modules/tool_parsing.py formats)
- strip markdown formatting (headings, emphasis, code fences; code *content*
  dropped entirely — same policy as extensions/silero_tts/tts_preprocessor.py,
  which is the starting point)
- collapse excess whitespace/newlines within a chunk
- voice events `(laugh)`, `(whispering)`, `(sigh)` etc. pass through
  verbatim — BreezeTTS2-native

## Worker (relay.py)

One worker thread per active conversation, sequential (TTS chunks within a
reply must play in order; audio.cpp may run other requests in parallel if the
user wants, but we stay sequential per stream).

Per queued chunk:
1. POST to audio.cpp `/v1/audio/speech` with the request body from
   stream_client.py:168-185, i.e.
   `{"model","voice","text","instruction","prompt_text"(auto from voice
   library),"temperature","top_p","top_k","min_p","seed","stream":true,
   "stream_frames_per_event","stream_lookahead_margin","text_chunk_mode",
   "text_chunk_size","response_format":"pcm"}`
2. Parse the upstream SSE (`stream_format: sse`): each `data:` line is a JSON
   `speech.audio.delta` with `audio` = base64 PCM16-LE mono frames (verified
   against app/server/runtime.cpp:2250-2254:
   `{"type":"speech.audio.delta","audio":"<b64>"}`; the AudioBuffer sample
   rate is the model's native rate — the delta does NOT carry it). Terminal
   event is `speech.audio.done` (runtime.cpp:2262-2264, with `timing`).
   The request body uses `input` (NOT `text`) for the prompt text, plus
   `options: {instruction, text_chunk_mode, text_chunk_size, temperature,
   top_p, top_k, min_p, stream_frames_per_event, stream_lookahead_margin,
   reference_text}` and top-level `model`, `voice`, `seed`,
   `response_format: "pcm"`, `stream: true`, `stream_format: "sse"`.
3. Relay each delta onto the browser SSE stream as
   `data: {"type":"audio","b64":...}`. Also append raw PCM to the
   server-side reassembly buffer (for the post-stream file).
4. On upstream error (`speech.error` / HTTP failure) emit
   `data: {"type":"error","message":...}` and continue with the next chunk.

Browser SSE framing (relay → client), one JSON per `data:` line:

```
{"type":"start","chat":...,"chunk":n,"text":...}     # chunk boundary (debug/UX)
{"type":"audio","b64":"..."}                          # PCM16 LE little-endian
{"type":"chunk_done","chunk":n}
{"type":"done","file":"file/..."|null}                # terminal; file = saved opus
{"type":"error","message":...}
```

**Stop behavior (user-confirmed: let it drain):** Stop/regenerate closes the
feeder; the worker finishes the in-flight TTS chunk, emits `done`, the relay
closes the client stream. No hard abort of upstream TTS.

**Browser disconnect:** relay detects broken pipe on the client SSE, marks
the stream stopped; the worker finishes the in-flight chunk (bounded) then
stops feeding; no new chunks are consumed.

## Player (player.js)

- On generation start for the current chat, `fetch('/voice/stream?chat=...',
  {headers})` and parse the SSE stream line-by-line (fetch, not EventSource,
  to allow an API key header if the relay ever needs one).
- WebAudio playback with the exact playhead-scheduling logic from
  stream_client.py:218-225: `nextTime = ctx.currentTime + 0.05` initial;
  `source.start(nextTime)`; `nextTime += buffer.duration`; never gap, never
  wait for buffer end.
- `AudioContext` sampleRate: the relay sends `sample_rate` in the `start`
  event (from the same setting used for server-side reassembly); the player
  creates the context at that rate (all modern browsers support arbitrary
  rates; Chrome caps at 96k — BreezeTTS2's 24k is fine).
- Stop button / stream `done` → finalize playhead, then swap in the saved
  file player (below).

## Post-stream file (opus)

- Server side reassembles all PCM16 of the reply → encode to **Opus in an Ogg
  container**. ENCODER CHOSEN AT BUILD TIME: `pyogg` 0.6.14a1 on this box has
  NO opus encoder (`PYOGG_OPUS_ENC_AVAIL=False`, sdist only, no wheel);
  `opuslib` 3.x is encoder-only (no Ogg muxer); **ffmpeg IS present
  (v8.0.1) and has a native `opus` encoder** → use ffmpeg via subprocess
  (`-f s16le -ar <rate> -i - -c:a libopus -b:a <kbps>k out.ogg`). Fallback
  chain: pyogg (if a future wheel exposes enc) → opuslib+manual ogg → WAV.
  → written to `extensions/audio_cpp/outputs/<chat>_<timestamp>.ogg`.
- Serving: the Gradio app serves the file at `GET /voice/file?path=<rel>`
  (validated, must live under the outputs dir) — so playback works regardless
  of what the custom gradio build exposes under `file/`. The terminal SSE
  `done` event carries `file: "/voice/file?path=..."` (relative to the
  Gradio origin)
  and JS appends `<audio controls src="<that>">` to the finished bot message
  so history stays replayable (silero_tts outputs/ pattern).
- Opus chosen over WAV/AAC: much smaller, fast encode, universal in-browser
  support; negligible compatibility cost (all target browsers play it).

## Configuration (settings.json keys, prefix `audio_cpp_`)

All user-configurable from the Extensions accordion:

Server:
- `server_url` — audio.cpp base URL, default `http://127.0.0.1:5023`
- `api_key` — optional bearer for the audio.cpp server
- `voice_dir` — voice library dir (`*.wav` + `prompt_text`), for voice
  dropdown + reference_text auto-fill (stream_client.py pattern)
- (no relay port — the /voice/* routes live on the Gradio app's own origin)

Model & sampling:
- `model` — audio.cpp model id (dropdown prefilled from `GET /v1/models` on
  "refresh models")
- `voice` — voice name (basename in the server voice dir); dropdown from
  `GET /v1/audio/voices?model=<id>` (audio.cpp runtime.cpp:3061 — returns a
  sorted name list from voice presets + embeddings/*.safetensors + voice_dir
  *.wav); free-text fallback when the endpoint is absent
- `prompt_text` — reference text; AUTO-filled from the voice library
  (voice → its `prompt_text`) when the user picks a voice (stream_client.py:262-276 pattern); overridable
- `instruction` — free-form per-reply instruction (sent as `instruction`)
- `temperature` (default 1.0), `top_p` (0.9), `top_k` (-1), `min_p` (0.05), `seed` (0 = random)

Streaming (upstream passthrough):
- `text_chunk_mode` (default `sentence`), `text_chunk_size` (default 250)
- `stream_frames_per_event` (default 200), `stream_lookahead_margin` (0 = server default)

Client-side chunking:
- `chunking_mode` — `sentence` (default) | `paragraph` | (custom, TBD)
- `min_chunk_chars` (120), `max_chunk_chars` (400)
- `split_punctuation` (default `.!?…:`)

Preprocessing toggles:
- `strip_thinking` (on), `strip_markdown` (on), `strip_tool_calls` (on),
  `collapse_whitespace` (on)

Output:
- `save_files` (on), `save_format` = `opus` (only option for now)
- `opus_bitrate_kbps` (64)
- `sample_rate` (default 24000 — VERIFY against the actual server model at
  implementation time; PCM frames are sample-rate-agnostic in transit but the
  server-side reassembly/encode needs it)

Behavior:
- `activate` (master toggle; off = modifier is a no-op passthrough)
- `bot_only` (on — voice name2 output only; user input never voiced)
- `request_timeout` (upstream TTS request timeout, default 120s)

## Files

```
extensions/audio_cpp/
├── DESIGN.md          # this file
├── script.py          # entry: setup/ui/custom_js/output_modifier + event handlers
├── chunker.py         # chunking strategies (sentence/paragraph/custom)
├── preprocessor.py    # thinking/markdown/tool/whitespace stripping
├── client.py          # audio.cpp HTTP client: models/voices discovery + SSE TTS
├── relay.py           # VoiceSession + worker threads + make_routes() (FastAPI router mounted on the Gradio app)
├── player.js          # browser SSE consumer + WebAudio playhead player
├── requirements.txt   # requests, pyogg
└── outputs/           # saved .ogg replies (gitignored)
```

`script.py` keeps the orchestration: settings load, conversation-key →
chunker/worker registry, output_modifier tap, Gradio accordion wiring,
"refresh models/voices", preview button (sends a test sentence through the
full pipeline).

## Verification plan

1. Unit: chunker (sentence boundaries, min/max coalescing, mid-word safety,
   flush-on-end, per-chat isolation) with a quick pytest-style script.
2. Unit: preprocessor against real thinking/tool/markdown samples.
3. Live: run audio.cpp (BreezeTTS2) on 5023, textgen with the extension on
   7860, generate a multi-paragraph reply; confirm audio starts well before
   the reply completes, no gaps in playback, `done` event, opus file created
   and playable from the message-embedded player.
4. Stop mid-generation → drains cleanly, no zombie worker, next reply reuses
   the worker fine.
5. Regenerate a reply → fresh stream, no contamination from previous chunks.

## Phase 2 (explicitly out of scope now)

audio.cpp running as a child process of textgen inside this extension
(spawn/stop from the accordion, log capture, port allocation). The design
already isolates all server contact in `client.py`, so phase 2 is a
lifecycle wrapper around it.

## Resolved open questions

1. **Tap = `custom_generate_reply`** (not `output_modifier`, which fires
   once at the end — text_generation.py:126-128). Wrapper re-yields each
   accumulated string unchanged and feeds the diff to the chunker.
2. **`sample_rate`**: NOT in the SSE stream (delta is `{"type", "audio"}`
   only, runtime.cpp:2250-2254). Comes from the `sample_rate` config
   setting (BreezeTTS2 = 24000; user-verified against their server).
3. **File serving**: the Gradio-mounted `/voice/file?path=` serves outputs/ —
   same-origin, so the `done` event's `file` URL is a relative path.
4. **Voices endpoint**: `GET /v1/audio/voices?model=<id>`
   (runtime.cpp:3061, `handle_voices`). Free-text fallback.

Also fixed vs. the initial draft: request field is `input` (not `text`);
opus encoding via **ffmpeg subprocess** (pyogg sdist has no encoder,
opuslib 3.x has no ogg muxer); `options` sub-object carries
instruction/text_chunk_mode/etc. (per stream_client.py:168-187, the
working client).
## Build status (2026-09-13)

Implemented and unit-verified:

- `chunker.py` + `tests/test_chunker.py` — ALL PASS (sentence boundaries,
  min/max coalescing, mid-word-safe hard split, flush-on-end, trailing-punct
  handling, per-chat isolation).
- `client.py` — models/voices discovery + SSE TTS streaming (`input` field,
  `options` sub-object, `response_format: pcm`, `stream_format: sse`).
- `relay.py` — ThreadingHTTPServer (CORS, superboogav2 pattern):
  `/voice/health`, `/voice/active` (newest in-flight session),
  `/voice/stream?chat=` (SSE), `/voice/models`, `/voice/voices`,
  `/voice/file?path=` (outputs dir, path-validated). Per-session worker
  thread with TWO queues (in_q chunks / out_q events — no worker-vs-reader
  race). Stop = drain, not abort. `done` event always emitted
  (worker wrapped in try/finally on `done_event`).
- `script.py` — `params` (is_tab: False → accordion in the shared
  extensions column), `setup()` (starts relay), `ui()` (accordion, all
  settings user-configurable; `.change` handlers persist to `audio_cfg`),
  `custom_generate_reply` tap (dispatch to generate_reply_HF/custom by model
  class exactly like text_generation.py:56; chat-mode only; re-yields
  byte-identical; feeds chunker per yield; GeneratorExit → cancel/drain;
  reap thread removes session after `done_event`).
- `player.js` — polls `/voice/active` every 250 ms; attaches to the live
  session SSE; WebAudio playhead scheduling (stream_client.py pattern,
  24 kHz AudioContext); on `done`, appends `<audio controls>` (the saved
  .ogg) to the newest bot message.
- `tests/test_relay_smoke.py` — mock audio.cpp (SSE deltas) end-to-end:
  client parsing, relay endpoints, session pipeline (5 events, audio then
  done), ffmpeg libopus encode to .ogg. ALL PASS.
- Full extension imports cleanly in the real textgen env
  (`importlib.import_module('extensions.audio_cpp.script')`).

Deltas vs. the design doc as built (all deliberate):
- `output_modifier` hook NOT used (no per-yield tap needed — the
  `custom_generate_reply` wrapper does everything).
- `api_key`/`bot_only`/`strip_citations` settings trimmed:
  relay host fixed to 127.0.0.1, no API key (local server), citations
  stripped by the markdown rule; `strip_tool_calls` folded into the
  thinking/tool strip.
- requirements.txt: no pip deps (stdlib + gradio; ffmpeg on PATH).
- **SSE keepalive (post-build fix, 2026-09-15)**: the first generation after a
  textgen restart always showed `[DONE] after 0 deltas` in the browser while
  the worker still relayed ~1000 events. Root cause: the SSE `gen()` used
  `out_q.get(timeout=15)` inside a `try` whose **blanket** `except Exception`
  swallowed `queue.Empty`. On a cold start the first LLM yield (engine warmup
  / first prefill) takes >15s, so the queue stayed empty, the exception fired,
  and the stream emitted a spurious `[DONE]`. The browser then latched
  `current` to that chat id and never re-attached. Fix: (1) catch
  `queue.Empty` separately and yield an SSE comment (`: keepalive`) so the
  connection survives quiet windows; (2) the remaining `except Exception`
  now logs `traceback` + re-raises (no fake `[DONE]`); (3) the player
  `allowReattach()` re-polls `/voice/active` on any close and re-subscribes
  if the session is still in-flight (up to 3 times).
- **Defaults**: `stream_frames_per_event=16`, `stream_lookahead_margin=12`
  (was 1/1) — matches audio.cpp's streaming tuning for low first-audio
  latency.
- **End-only thinking suppression (post-build fix, 2026-09-15)**: some models
  have their *opening* thinking tag embedded in the instruction template, so
  the raw stream only ever carries the *closing* tag. Until that tag appears,
  `modules.reasoning.extract_reasoning` cannot know the text is thinking (it
  returns it as final), so the whole reasoning block was chunked and spoken —
  and when the reasoning quotes the reply string, that sounded like the reply
  "repeating" (281 thinking-chunks observed). Fix: at session creation the
  extension scans `state['instruction_template_str' / 'chat_template_str']`
  for tags from `modules.reasoning.THINKING_FORMATS` (embedded opener ⇒ its
  closer; bare closer ⇒ end-only model); while the detected closer is absent
  from the cumulative stream, `clean()` suppresses the text entirely, and
  once it arrives, standard extraction splits at it. A "Thinking end-tag
  (override)" settings field (blank = auto-detect) forces a tag; a
  "thinking-end-tag never appeared — nothing voiced" warning covers the
  skipped-thinking case.
- `specs.py` + `model_specs/` (post-build fix): the server validates every
  `options` key against the model-contract `request_option_keys`
  (spec_backed_model.h:59) and rejects the WHOLE request with "unknown
  <family> request option" (no HTTP endpoint exposes the spec). `model_specs/`
  holds EXACT COPIES of the audio.cpp in-repo `model_specs/*.json` (77 files,
  Apache-2.0 — see `model_specs/AUDIO_CPP_LICENSE`; copied 2026-09-14) so the
  user can edit/replace them to match their server build. `specs.py` loads
  them at import: per-family allow-list from `options.request[].name`, family
  detected from the model id by filename heuristic (the /v1/models response
  carries no family field; a GGUF may embed a newer spec, so unknown families
  pass through unfiltered). `client.py` drops rejected keys before sending
  (logged). Model dropdown labels are `id  [family]`; bare ids are stored.

To run: `python server.py --extensions audio_cpp` (plus whatever LLM
backend), then in the browser hit the "Audio CPP" accordion, set
server_url/model/voice, tick Enable, generate a chat reply.
