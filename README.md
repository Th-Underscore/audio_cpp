# audio.cpp TTS Extension

Streams TTS for text generated in oobabooga's text-generation-webui, synthesized by an external [audio.cpp](https://github.com/theroar001/audio.cpp) server. Audio starts ~1–2 sentences into a reply, not after it.

## Quick features

- Streams audio while text is produced by the model (sentence/paragraph chunking mid-generation)
- Toggle on/off from the UI; finished replies are saved as `.ogg` and appended to the message
- Per-model synthesis options (temperature, top_k/p, guidance, streaming, …) with a **Save** button — persisted to `config.json` and reloaded at startup
- Voice library auto-fill: pick a voice (`*.wav` + `prompt_text` in the voice dir) and `reference_text` is filled from the prompt
- Any audio.cpp family supported (`/v1/models` drives the dropdown) — TTS, music, effects, …

## Requirements

- An audio.cpp server you run yourself (a separate process; this extension only talks to it over HTTP). Default URL `http://127.0.0.1:5023`.
- ffmpeg on `PATH` (`.pcm` → `.ogg` encoding of saved replies)
- `pip install -r extensions/audio_cpp/requirements.txt`

## Installation

```bash
cd path/to/text-generation-webui/extensions
git clone https://github.com/Th-Underscore/audio_cpp.git
```

## Usage

1. **Start the audio.cpp server first.** The extension's discovery hooks run when the web UI opens, so if the server isn't up yet the model/voice dropdowns will be empty. Once it's running, click **Refresh models & voices** in the accordion — it re-fetches the server live.

2. Launch the web UI with the extension:

    ```bash
    cd path/to/text-generation-webui
    python server.py --extensions audio_cpp
    ```

2. Open "Text generation", expand the **audio.cpp TTS** accordion:
    - set the server URL, then **Refresh models & voices**
    - pick a TTS model + voice
    - toggle **Enable voicing of bot replies**
3. Generate — audio streams in and the saved `.ogg` appears on the bot message

## Configuration

The accordion groups settings into **global** (connection, client-side chunking, text preprocessing, saving) and **Synthesis options (per-model)** — the latter are keyed by the selected model id, so each model keeps its own sampling/delivery settings.

- **Save** (primary button) writes `extensions/audio_cpp/config.json`; it is loaded automatically at startup
- Switching the TTS model dropdown reloads that model's saved synthesis options
- "Thinking close-tag override" empty = auto-detected from the loaded LLM's reasoning format
- `DESIGN.md` documents the pipeline, hooks, and implementation details

## To-Do

- Server startup: a UI control (button/toggle in the accordion) that starts the audio.cpp server process itself, so the web UI no longer has to be opened after the server — the extension would own the server lifecycle (spawn `server -m ...`, wait for `/v1/models`, stop on UI close)