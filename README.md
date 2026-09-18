# audio.cpp TTS Extension

Streams TTS for text generated in oobabooga's text-generation-webui, synthesized by an external [audio.cpp](https://github.com/0xShug0/audio.cpp) server. Audio starts ~1–2 sentences into a reply, not after it.

## Quick features

- Streams audio while text is produced by the model (sentence/paragraph chunking mid-generation)
- Toggle on/off from the UI; finished replies are saved as `.opus` and appended to the bot message
- **Stop voice** button (floating, bottom-right) appears while audio is in flight — hard-interrupts the in-flight synthesis (drops the audio.cpp connection and queued chunks) and stops playback
- Per-model synthesis options (temperature, top_k/p, guidance, streaming, …) with a **Save** button — persisted to `config.json` and reloaded at startup
- Voice library auto-fill: pick a voice (`*.wav` + `prompt_text` in the voice dir) and `reference_text` is filled from the prompt
- Any audio.cpp family supported (`/v1/models` drives the dropdown) — TTS, music, effects, …

## Requirements

- An audio.cpp server you run yourself (a separate process; this extension only talks to it over HTTP). Default URL `http://127.0.0.1:5023`.
- ffmpeg on `PATH` — only to encode **saved** replies (`.pcm` → `.opus`).
- The **live** Opus stream needs **no ffmpeg and no subprocess**: it's encoded in-process via `ctypes`-bound **libopus** plus a pure-Python Ogg page writer. The only install is the **libopus shared library** on the system (e.g. `libopus0` on Debian/Ubuntu, usually already present; `opus.dll`/`libopus-0.dll` on Windows). The loader tries `opus` / `libopus.so.0` / `libopus.so` / `libopus-0.dll` in turn, overridable with the `AUDIOCPP_LIBOPUS` env var (full path to the `.so`/`.dll`). If libopus isn't found the relay falls back to the PCM transport.
- `pip install -r extensions/audio_cpp/requirements.txt`

### Building audio.cpp from source

Not required - just what I did. Build the CUDA CLI + server targets (V100 = arch 70), then serve the binary:

```bash
devbuild -r 10g nvidia/cuda:12.9.0-devel-ubuntu24.04 \
  "scripts/build_linux.sh --backend cuda --target audiocpp_server --cuda-arch 70"

./build/linux-cuda-release/bin/audiocpp_server
```

## Installation

```bash
cd path/to/text-generation-webui/extensions
git clone https://github.com/Th-Underscore/audio_cpp.git
```

## Usage

1. **Start the audio.cpp server first.** The extension's discovery hooks run when the web UI opens, so if the server isn't up yet the model/voice dropdowns will be empty (static Gradio elements). Once it's running, click **Refresh models & voices** in the accordion — it re-fetches the server live.

2. Launch the web UI with the extension:

    ```bash
    cd path/to/text-generation-webui
    python server.py --extensions audio_cpp
    ```

2. Open "Text generation", expand the **audio.cpp TTS** accordion:
    - Set the server URL, then **Refresh models & voices**
    - Pick a TTS model + voice
    - Toggle **Enable voicing of bot replies**
3. Generate — audio streams in and the saved `.opus` appears on the bot message
4. Mid-synthesis, **Stop voice** aborts the TTS immediately (vs. textgen's Stop/Regenerate, which let the in-flight chunk drain)

## Configuration

The accordion groups settings into **global** (connection, client-side chunking, text preprocessing, saving) and **Synthesis options (per-model)** — the latter are keyed by the selected model id, so each model keeps its own sampling/delivery settings.

- **Save** (primary button) writes `extensions/audio_cpp/config.json`; it is loaded automatically at startup
- Switching the TTS model dropdown reloads that model's saved synthesis options
- "Thinking close-tag override" empty = auto-detected from the loaded LLM's reasoning format


## To-Do

- Better CSS styling
- `ffmpeg`/`libopus` in extension root
- Direct support for other models
- Server startup: a UI control (button/toggle in the accordion) that starts the audio.cpp server process itself, so the web UI no longer has to be opened after the server — the extension would own the server lifecycle (spawn `audiocpp_server -m ...`, wait for `/v1/models`, stop on UI close)
- "Notebook" for manually-triggered synthesis
- Expression AI-rewrite: Insert configurable (expression) tags via LLM per chunk
- Improved silence padding logic: Noise floor, silence detection -> pad timing, etc.
- Proper refreshing (currently updates server-side, but if audio.cpp server isn't listening during boot, TTS model id and Voice never get their values client-side)
- STT
- Non-UI-dependent API
- Per-token streaming for models that support it (like [VibeVoice Realtime TTS](https://github.com/Th-Underscore/vibevoice_realtime)) \[currently unavailable in audio.cpp itself\]
