"""audio.cpp TTS extension (server-side streaming, no extra port).

Voices bot replies through any audio.cpp server. Topology:

    browser  <-SSE-  this Gradio app /voice/*  <-POST/SSE-  audio.cpp
                             ^
                             | TextChunker
                             |
                     custom_generate_reply wrapper (per-yield text feed)

All querying of audio.cpp happens server-side; the browser only talks to the
Gradio app it's already connected to. Routes are mounted on the Gradio
FastAPI app by a deferred thread (the app object is created in queue(),
after extension setup() has run).
"""

import os
import json
import re
import time
import uuid
import threading
import traceback
from collections import OrderedDict

import gradio as gr
import modules.shared as shared

from . import client
from . import relay
from . import specs

params = {
    "display_name": "Audio CPP",
    "is_tab": False,  # accordion in the shared extensions column
}

# ---------------------------------------------------------------------------
# Extension settings
#
# The config has two layers:
#   GLOBAL   - connection + client-side behaviour, shared by every model
#              (enabled, server url, voice dir, selected model + voice, sample
#               rate, request timeout, client-side chunking, text
#               preprocessing, file saving, thinking-tag override).
#   PER-MODEL - the "Synthesis options" sent to audio.cpp on every request,
#              keyed by the CONCRETE model id (not the family), so each loaded
#              model keeps its own sampling/delivery settings.
# `audio_cfg` is the flat in-memory view: GLOBAL merged with the currently
# selected model's PER-MODEL options. relay.py and the TTS tap read it at
# runtime.
# ---------------------------------------------------------------------------
MODEL_OPTION_KEYS = [
    "temperature", "depth_temperature", "top_k", "top_p", "min_p",
    "guidance_scale", "max_tokens", "seed", "instruction",
    "reference_text", "text_chunk_mode", "text_chunk_size",
    "stream_frames_per_event", "stream_lookahead_margin",
]

_DEFAULT_GLOBAL = {
    "enabled": False,
    "server_url": "http://127.0.0.1:5023",
    "voice_dir": "",
    "model": "",
    "voice": "",
    "sample_rate": 24000,
    "request_timeout": 120,
    "chunk_mode": "sentence",
    "chunk_min_chars": 120,
    "chunk_max_chars": 400,
    "strip_thinking": True,
    "strip_markdown": True,
    "strip_citations": True,
    "save_file": True,
    "thinking_end_tag": "",
}

_DEFAULT_MODEL_OPTS = {
    "temperature": 0.9,
    "depth_temperature": 0.9,
    "top_k": 0,
    "top_p": 0.9,
    "min_p": 0.0,
    "guidance_scale": 1.0,
    "max_tokens": 1500,
    "seed": -1,
    "instruction": "",
    "reference_text": "",
    "text_chunk_mode": "default",
    "text_chunk_size": 600,
    "stream_frames_per_event": 16,
    "stream_lookahead_margin": 12,
}

_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "config.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

audio_cfg = dict(_DEFAULT_GLOBAL)
audio_cfg.update(_DEFAULT_MODEL_OPTS)
_model_options = {}   # bare model id -> {synthesis option: value}

_sessions = OrderedDict()   # stream_id -> VoiceSession
_sessions_lock = threading.Lock()
_current_stream = None       # stream_id being generated RIGHT NOW
_registry_lock = threading.Lock()
_mounted = threading.Event()
_model_families = {}        # model id -> family (discovery cache)


def _set_setting(key, value):
    audio_cfg[key] = value
    return value


def _model_opts_for(model_id):
    """The active model's stored synthesis options, with defaults for any
    missing keys (so a model saved with a partial set still renders fully)."""
    base = dict(_DEFAULT_MODEL_OPTS)
    stored = _model_options.get(model_id) if model_id else None
    if isinstance(stored, dict):
        base.update({k: stored[k] for k in MODEL_OPTION_KEYS if k in stored})
    return base


def _load_config():
    """Load config.json at startup, if present.

    Shape: {"global": {...}, "models": {<model id>: {synthesis options}}}.
    Tolerant: only known global keys are applied and each model entry is
    restricted to the model-option keys, so a hand-edited or older file
    cannot corrupt the in-memory config.
    """
    global _model_options
    if not os.path.exists(_SETTINGS_FILE):
        return
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception as e:
        shared.logger.warning("audio_cpp: failed to load config.json: %s" % e)
        return
    if not isinstance(saved, dict):
        return
    g = saved.get("global")
    if isinstance(g, dict):
        for k, v in g.items():
            if k in _DEFAULT_GLOBAL:
                audio_cfg[k] = v
    models = saved.get("models")
    if isinstance(models, dict):
        _model_options = {}
        for mid, opts in models.items():
            if isinstance(opts, dict):
                _model_options[str(mid)] = {
                    k: opts[k] for k in MODEL_OPTION_KEYS if k in opts}
    # Re-apply the currently selected model's options on top of global so the
    # flat view matches what was saved.
    audio_cfg.update(_model_opts_for(audio_cfg.get("model", "")))
    shared.logger.info(
        "audio_cpp: loaded config.json (model=%r, %d per-model option sets)"
        % (audio_cfg.get("model"), len(_model_options)))


# ---------------------------------------------------------------------------
# Discovery (Gradio endpoint handlers)
# ---------------------------------------------------------------------------
def _model_choices(server_url):
    """Model ids as dropdown labels 'id  [family]'; also cache the families
    (audio.cpp /v1/models carries `family`; specs.detect_family is the
    fallback) — the speech request options get filtered per family because
    the server rejects any key outside the model's spec allow-list."""
    global _model_families
    try:
        models = client.get_models(server_url)
    except Exception as e:
        shared.logger.warning("audio_cpp model discovery failed: %s", e)
        return []
    _model_families = client.get_model_families(models)
    return ["%s  [%s]" % (i, f) if f else i for i, f in models]


def _voice_choices(server_url, model):
    names = []
    try:
        names = client.get_voices(server_url, _bare_model_id(model))
    except Exception as e:
        shared.logger.warning("audio_cpp voice discovery failed: %s", e)
    # merge the local voice library (stream_client.py pattern)
    lib, _ = relay.list_voice_library(audio_cfg.get("voice_dir", ""))
    seen, out = set(), []
    for n in names + lib:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _bare_model_id(label):
    """Dropdown label 'id  [family]' -> bare id; plain ids pass through."""
    if label is None:
        return ""
    label = str(label).strip()
    if label.endswith("]") and " [" in label:
        label = label.rsplit(" [", 1)[0].strip()
    return label


def _model_value(choices, model_id):
    """Map a saved bare model id (config.json 'model') to its dropdown label
    (choices are 'id  [family]'); None when the model is not among the
    discovered choices. Gradio requires value to exactly match a choice,
    so the raw id alone would render the dropdown empty."""
    if not model_id:
        return None
    model_id = str(model_id).strip()
    for c in choices:
        if c == model_id or _bare_model_id(c) == model_id:
            return c
    return None


def _voice_transcript(voice):
    _, texts = relay.list_voice_library(audio_cfg.get("voice_dir", ""))
    return texts.get(voice, "")


def _refresh(server_url, voice_dir, model):
    """Refresh button: rediscover models + voices, return both choices."""
    _set_setting("server_url", server_url)
    _set_setting("voice_dir", voice_dir)
    models = _model_choices(server_url)
    voices = _voice_choices(server_url, _bare_model_id(model))
    return models, voices


# ---------------------------------------------------------------------------
# Deferred Gradio route mounting
# ---------------------------------------------------------------------------
def _mount_loop():
    """Mount /voice/* on the Gradio app — and re-mount whenever it is replaced.

    Gradio creates blocks.app THREE times: Blocks.__exit__ (end of the
    `with gr.Blocks()` in server.py), queue(), and launch() — each call
    REPLACES self.app with a fresh FastAPI. The served app is the one
    launch() builds, so we must track the object identity and re-include
    the router whenever it changes.
    """
    from fastapi import FastAPI
    shared.logger.info("audio_cpp: mount loop started (waiting for gradio app)")
    mounted_on = None
    while True:
        try:
            blocks = shared.gradio.get("interface")
            app = getattr(blocks, "app", None)
        except Exception:
            app = None
        if app is not None:
            try:
                if isinstance(app, FastAPI) and app is not mounted_on:
                    router = relay.make_routes(
                        _sessions, lambda: dict(audio_cfg))
                    # Registry routes added here (they need script.py scope)
                    # BEFORE include_router: include_router copies the route
                    # list at call time, so routes added after it would
                    # exist on the router but never on the live app.
                    router.get("/voice/registry")(_registry_route)
                    router.delete("/voice/registry")(_registry_delete_route)
                    app.include_router(router)
                    mounted_on = app
                    shared.logger.info(
                        "audio_cpp /voice/* routes mounted on %r" % (app,))
                _mounted.set()
            except Exception:
                traceback.print_exc()
        time.sleep(0.5)


def setup():
    # wire the debug sinks to textgen's log (visible in the server console)
    client.set_logger(lambda m: shared.logger.info("audio_cpp: " + m))
    relay.set_logger(lambda m: shared.logger.info("audio_cpp: " + m))
    specs.set_logger(lambda m: shared.logger.info("audio_cpp: " + m))
    # Register this module as a loaded extension. load_extensions() normally
    # does this, but only when the extension is in shared.args.extensions;
    # when enabled another way, history_modifier/custom_generate_reply would
    # never be dispatched without the state entry.
    import sys as _sys
    import modules.extensions as _ext
    if 'audio_cpp' not in _ext.state:
        _ext.state['audio_cpp'] = [True, len(_ext.state), _sys.modules[__name__]]
    # Load the persisted voice registry once (idempotent).
    _load_registry()
    if not _mounted.is_set():
        threading.Thread(target=_mount_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# custom_generate_reply — the stream tap
# ---------------------------------------------------------------------------
def _thinking_end_tag(state, question):
    """Thinking close tag for the active model + this generation.

    User override wins. Otherwise render the ACTUAL prompt via textgen's own
    generate_chat_prompt() with a throwaway single-turn history (so no
    earlier turns can leak unclosed tags in), with the ACTIVE settings —
    Jinja conditionals like ``enable_thinking`` are resolved — and apply the
    "last opened tag closed within the prompt?" rule:
      - last opener left UNCLOSED in the prompt  -> model is in thinking
        mode; suppress the stream until its close tag appears;
      - last opener CLOSED in the prompt (the
        enable_thinking=false rendering emits the pair) -> normal stream;
      - no opener, bare close tag in prompt -> end-only model.
    Falls back to the raw-template scan if rendering fails.
    """
    override = (audio_cfg.get("thinking_end_tag") or "").strip()
    if override:
        return override
    try:
        from . import preprocessor
        from modules.chat import generate_chat_prompt
        fake_history = {
            "pairs": [["[[Placeholder]]", ""]],
            "internal": [["<|BEGIN-VISIBLE-CHAT|>", "[[Placeholder]]"]],
            "metadata": {},
        }
        rendered = generate_chat_prompt(question, state, history=fake_history)
        tag = preprocessor.detect_end_tag_rendered(rendered)
        if tag:
            return tag
        # Rendered prompt shows no thinking at all (and no bare close tag):
        # nothing to suppress — do NOT fall through to the raw scan, which
        # would see tags in inactive conditional branches.
        return ""
    except Exception:
        pass
    try:
        from . import preprocessor
        tmpl = state.get("instruction_template_str", "") or ""
        if not tmpl:
            tmpl = state.get("chat_template_str", "") or ""
        return preprocessor.detect_end_tag(tmpl)
    except Exception:
        return ""


def _chat_id(state):
    try:
        return str(state['unique_id'])
        # ch = getattr(state, "chat", None)
        # if isinstance(ch, dict) and ch.get("id"):
        #     return str(ch["id"])
    except Exception:
        pass
    return "nochat"


def custom_generate_reply(question, original_question, state, stopping_strings,
                          is_chat):
    if not (audio_cfg.get("enabled") and audio_cfg.get("model")):
        shared.logger.info("audio_cpp: tap: PASS-THROUGH (enabled=%s model=%r)"
                           % (audio_cfg.get("enabled"), audio_cfg.get("model")))
        yield from _base_reply(question, original_question, state,
                                stopping_strings, is_chat)
        return

    if not is_chat:
        # Only voice chat-mode bot replies (not raw prompt completion).
        shared.logger.info("audio_cpp: tap: PASS-THROUGH (not chat mode)")
        yield from _base_reply(question, original_question, state,
                                stopping_strings, is_chat)
        return

    _prune_sessions()
    stream_id = _new_stream_id()
    end_tag = _thinking_end_tag(state, question)
    with _sessions_lock:
        _sessions[stream_id] = relay.VoiceSession(
            stream_id, dict(audio_cfg), on_file=_pending_file_cb,
            thinking_end_tag=end_tag)
        global _current_stream
        _current_stream = stream_id
    sess = _sessions[stream_id]
    shared.logger.info("audio_cpp: tap: VOICING stream_id=%s chat=%s "
                        "end_tag=%r q=%r"
                        % (stream_id, _chat_id(state), end_tag,
                           (question or "")[:80]))

    n_yield = 0
    total_len = 0
    try:
        for output in _base_reply(question, original_question, state,
                                   stopping_strings, is_chat):
            n_yield += 1
            total_len = len(output)
            sess.feed(output)
            yield output
    except GeneratorExit:
        # Stop button: drain the in-flight synthesis, don't cut it off.
        shared.logger.info("audio_cpp: tap: GeneratorExit after %d yields "
                           "(last len=%d), cancelling" % (n_yield, total_len))
        sess.cancel()
        raise
    finally:
        shared.logger.info("audio_cpp: tap: base stream ended: %d yields, "
                           "final len=%d" % (n_yield, total_len))
        sess.finish()
        threading.Thread(target=_reap, args=(stream_id, sess, state),
                         daemon=True).start()


def _reap(stream_id, sess, state):
    """Worker thread: wait for synthesis+opus-save to finish, then record
    the finished file in the extension-local registry (see _record_audio).
    The .ogg is only on disk once the relay worker's on_file callback
    fires, which is AFTER the tap's finally runs — so the recording is
    done HERE, not in the tap."""
    try:
        while not sess.done_event.is_set():
            time.sleep(0.2)
    except Exception:
        pass
    _record_audio(stream_id, state)
    with _sessions_lock:
        _sessions.pop(stream_id, None)


# ---------------------------------------------------------------------------
# Voice registry — audio files keyed per message row (client overlay)
# ---------------------------------------------------------------------------
# The extension keeps its own voice registry (textgen core stays untouched):
#   {"assistant_<idx>": {"path": <relpath vs OUT_DIR>, "ts": ...}}
# persisted in audio_registry.json next to this file. The client mirrors it
# into localStorage (player.js) and anchors a chip to each matching
# .message[data-index]. Regenerating a message creates a NEW row (new idx),
# so a new synthesis records a NEW key — files never stack on one row. The
# displayed version is read from the DOM for chip labeling only.
REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "audio_registry.json")
_voice_registry = {}


def _load_registry():
    global _voice_registry
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _voice_registry = data
    except Exception:
        _voice_registry = {}


def _save_registry():
    try:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(_voice_registry, f, indent=2, ensure_ascii=False)
    except Exception:
        traceback.print_exc()


def get_registry() -> dict:
    """Snapshot of the registry (served by GET /voice/registry)."""
    with _registry_lock:
        return {k: dict(v) for k, v in _voice_registry.items()}


def _registry_route():
    return get_registry()


def _registry_delete_route(key: str = ""):
    """DELETE /voice/registry?key=assistant_<uid>_<idx>_<v>

    The registry is per (chat, row, version). Also accepts legacy bare
    assistant_<idx> keys from old registry files. The client sends the
    exact key it is anchoring to. Returns {"ok": true, "deleted": <bool>}"""
    k = (key or "").strip()
    if re.match(r"^assistant_\d+(?:_\d+)?$", k) or \
            re.match(r"^assistant_.+?_\d+_\d+$", k):
        with _registry_lock:
            deleted = _voice_registry.pop(k, None) is not None
            if deleted:
                _save_registry()
        return {"ok": True, "deleted": deleted}
    return {"ok": False, "error": "bad key", "deleted": False}


def _record_audio(stream_id, state):
    """Record the stream's finished .ogg under the message row the session
    belonged to: assistant_<uid>_<idx>_<v>.

    uid = chat unique id (state['unique_id'] — same value the #past-chats
    Radio and viewing_unique_id use), so audio never bleeds across chats.
    v = number of versions of that row at recording time: regeneration
    appends the new version before the stream finishes, so v is exactly the
    version just synthesized — each version of a row keeps its own file
    and regenerates add, never overwrite."""
    if not audio_cfg.get("save_file", True):
        return
    try:
        file_path = None
        with _sessions_lock:
            sess = _sessions.get(stream_id)
        if sess is not None and getattr(sess, "file_path", None):
            file_path = sess.file_path
        if not file_path:
            return
        if not os.path.exists(file_path):
            shared.logger.warning("audio_cpp: pending file missing: %s"
                                   % file_path)
            return
        history = state.get("history") if isinstance(state, dict) else None
        if not history:
            return
        idx = len(history.get("internal", [])) - 1
        if idx < 0:
            return
        uid = _chat_id(state)
        versions = (history.get("metadata", {}) or {}).get(
            "assistant_%d" % idx, {}).get("versions", [])
        v = max(len(versions), 1)
        rel = os.path.relpath(file_path, relay.OUT_DIR)
        key = "assistant_%s_%d_%d" % (uid, idx, v)
        with _registry_lock:
            _voice_registry[key] = {"path": rel, "ts": time.time()}
            _save_registry()
        shared.logger.info("audio_cpp: recorded audio %s -> %s"
                           % (rel, key))
    except Exception:
        traceback.print_exc()


def _new_stream_id():
    return uuid.uuid4().hex[:12]


def _prune_sessions(max_age=300.0, cap=200):
    now = time.time()
    with _sessions_lock:
        stale = [k for k, s in _sessions.items() if now - s.ts > max_age]
        for k in stale:
            _sessions.pop(k, None)
        while len(_sessions) > cap:
            _sessions.popitem(last=False)


def _base_reply(question, original_question, state, stopping_strings, is_chat):
    from modules import text_generation
    # Same dispatch as text_generation._generate_reply:56 (textgen picks the
    # backend by model class name, not by a 'model_mode' key).
    model_cls = shared.model.__class__.__name__ if shared.model is not None else None
    if model_cls in ['LlamaServer', 'Exllamav3Model', 'TensorRTLLMModel',
                     'LMDeployModel']:
        gen = text_generation.generate_reply_custom
    else:
        gen = text_generation.generate_reply_HF
    shared.logger.info("audio_cpp: base dispatch model_class=%s -> %s (is_chat=%s)"
                       % (model_cls, gen.__name__, is_chat))
    return gen(question, original_question, state, stopping_strings,
               is_chat=is_chat)

def _pending_file_cb(file_path):
    """VoiceSession.on_file: the .ogg is on disk. Stashed on the session so
    the reaper (_record_audio) can pick it up AFTER done_event is set."""
    global _current_stream
    with _sessions_lock:
        sess = _sessions.get(_current_stream) if _current_stream else None
    if sess is not None:
        sess.file_path = file_path
    shared.logger.info("audio_cpp: file saved %s (stream %s)"
                       % (os.path.basename(file_path), _current_stream))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def ui():
    with gr.Accordion("audio.cpp TTS", open=False) as acc:
        # -- global (shared by every model) ---------------------------------
        enabled = gr.Checkbox(value=audio_cfg["enabled"],
                              label="Enable voicing of bot replies")
        server_url = gr.Textbox(value=audio_cfg["server_url"],
                                label="audio.cpp server URL")
        voice_dir = gr.Textbox(value=audio_cfg.get("voice_dir", ""),
                               label="Voice library dir (/*.wav + prompt_text)")
        model_choices = _model_choices(audio_cfg["server_url"])
        model = gr.Dropdown(choices=model_choices,
                            value=_model_value(model_choices,
                                               audio_cfg["model"]),
                            label="TTS model id (from /v1/models)")
        voice = gr.Dropdown(choices=_voice_choices(
            audio_cfg["server_url"], audio_cfg["model"] or None),
            value=audio_cfg["voice"] or None,
            label="Voice (server + voice library)")
        refresh = gr.Button("Refresh models & voices")
        with gr.Row():
            sample_rate = gr.Number(value=audio_cfg["sample_rate"],
                                    label="Sample rate (server PCM output)",
                                    precision=0)
            request_timeout = gr.Number(value=audio_cfg["request_timeout"],
                                         label="TTS request timeout (s)",
                                         precision=0)

        # -- synthesis options (PER-MODEL) -----------------------------------
        gr.Markdown("### Synthesis options (per-model)")
        temperature = gr.Slider(0.0, 2.0, value=audio_cfg["temperature"],
                                label="temperature")
        depth_temperature = gr.Slider(0.0, 2.0,
                                       value=audio_cfg["depth_temperature"],
                                       label="depth_temperature")
        top_k = gr.Number(value=audio_cfg["top_k"], label="top_k",
                          precision=0)
        top_p = gr.Slider(0.0, 1.0, value=audio_cfg["top_p"], label="top_p")
        min_p = gr.Slider(0.0, 1.0, value=audio_cfg["min_p"], label="min_p")
        guidance_scale = gr.Slider(0.0, 10.0,
                                   value=audio_cfg["guidance_scale"],
                                   label="guidance_scale")
        max_tokens = gr.Number(value=audio_cfg["max_tokens"],
                               label="max_tokens", precision=0)
        seed = gr.Number(value=audio_cfg["seed"], label="seed", precision=0)
        instruction = gr.Textbox(value=audio_cfg["instruction"],
                                  lines=2,
                                  label="instruction (style/delivery)")
        reference_text = gr.Textbox(value=audio_cfg["reference_text"],
                                     lines=2,
                                     label="reference_text (auto-filled from "
                                           "voice library; overridable)")
        text_chunk_mode = gr.Textbox(value=audio_cfg["text_chunk_mode"],
                                      label="audio.cpp text_chunk_mode")
        text_chunk_size = gr.Number(value=audio_cfg["text_chunk_size"],
                                     label="audio.cpp text_chunk_size",
                                     precision=0)
        stream_frames_per_event = gr.Number(
            value=audio_cfg["stream_frames_per_event"],
            label="stream_frames_per_event", precision=0)
        stream_lookahead_margin = gr.Number(
            value=audio_cfg["stream_lookahead_margin"],
            label="stream_lookahead_margin", precision=0)

        # synthesis widgets in the SAME order as MODEL_OPTION_KEYS — the model
        # change handler returns values in this order to reload them.
        synth_inputs = [temperature, depth_temperature, top_k, top_p, min_p,
                        guidance_scale, max_tokens, seed, instruction,
                        reference_text, text_chunk_mode, text_chunk_size,
                        stream_frames_per_event, stream_lookahead_margin]

        # -- chunking (client-side, global) ----------------------------------
        gr.Markdown("### Chunking (client-side)")
        chunk_mode = gr.Dropdown(["sentence", "paragraph"],
                                  value=audio_cfg["chunk_mode"],
                                  label="Chunk mode (client-side)")
        chunk_min_chars = gr.Number(value=audio_cfg["chunk_min_chars"],
                                     label="Min chars per chunk", precision=0)
        chunk_max_chars = gr.Number(value=audio_cfg["chunk_max_chars"],
                                     label="Max chars per chunk", precision=0)

        # -- text preprocessing (global) -------------------------------------
        gr.Markdown("### Text preprocessing (before TTS)")
        strip_thinking = gr.Checkbox(value=audio_cfg["strip_thinking"],
                                      label="Strip thinking/reasoning")
        strip_markdown = gr.Checkbox(value=audio_cfg["strip_markdown"],
                                      label="Strip markdown/formatting")
        strip_citations = gr.Checkbox(value=audio_cfg["strip_citations"],
                                       label="Strip [1]-style citations")
        save_file = gr.Checkbox(value=audio_cfg["save_file"],
                                 label="Save finished reply as .ogg")
        thinking_end_tag = gr.Textbox(
            value=audio_cfg["thinking_end_tag"],
            label="Thinking close-tag override (empty = auto-detect)",
            interactive=True,
            elem_classes=["audio_cpp_narrow"])

        save = gr.Button("Save", variant="primary")
        status = gr.Markdown("")

        # Global + synthesis fields: keep the in-memory audio_cfg in sync so
        # the live tap reflects edits immediately. (Disk persistence is the
        # Save button's job.)
        global_map = {
            enabled: "enabled", server_url: "server_url",
            voice_dir: "voice_dir", voice: "voice",
            sample_rate: "sample_rate", request_timeout: "request_timeout",
            chunk_mode: "chunk_mode", chunk_min_chars: "chunk_min_chars",
            chunk_max_chars: "chunk_max_chars",
            strip_thinking: "strip_thinking", strip_markdown: "strip_markdown",
            strip_citations: "strip_citations", save_file: "save_file",
            thinking_end_tag: "thinking_end_tag",
        }
        for el, key in global_map.items():
            el.change(_field_updater(key), el)
        synth_map = dict(zip(synth_inputs, MODEL_OPTION_KEYS))
        for el, key in synth_map.items():
            el.change(_field_updater(key), el)

        # Switching the TTS model loads THAT model's stored synthesis options
        # into the widgets (and into audio_cfg) — this is what makes them
        # per-model.
        model.change(_load_model_opts, model, synth_inputs)

        # model + voice dropdowns: rediscover choices
        refresh.click(_refresh, [server_url, voice_dir, model],
                      [model, voice])
        model.select(_voice_choices, [server_url, model], [voice])
        # auto-fill reference_text from the voice library
        voice.change(_voice_transcript, voice, reference_text)

        # Persist the whole config (global + every known per-model option set,
        # with the current model's live values) to config.json.
        save.click(_persist, None, status)

        return acc


# Coercion per setting key (kept in one place for the field-update closures
# and the config save).
_COERCERS = {
    "enabled": bool, "server_url": str, "voice_dir": str, "model": str,
    "voice": str, "sample_rate": int, "request_timeout": int,
    "chunk_mode": str, "chunk_min_chars": int, "chunk_max_chars": int,
    "strip_thinking": bool, "strip_markdown": bool, "strip_citations": bool,
    "save_file": bool, "thinking_end_tag": str,
    "temperature": float, "depth_temperature": float, "top_k": int,
    "top_p": float, "min_p": float, "guidance_scale": float,
    "max_tokens": int, "seed": int, "instruction": str,
    "reference_text": str, "text_chunk_mode": str, "text_chunk_size": int,
    "stream_frames_per_event": int, "stream_lookahead_margin": int,
}


def _coerce(key, value):
    return _COERCERS[key](value)


def _field_updater(key):
    """Return a handler that copies one widget's value into the in-memory
    audio_cfg (coerced) so the running tap sees edits immediately. Disk
    persistence is handled separately by the Save button."""
    def update(value):
        try:
            _set_setting(key, _coerce(key, value))
        except (TypeError, ValueError):
            pass
    return update


def _load_model_opts(model_label):
    """Model dropdown changed: load that model's stored synthesis options
    (defaults for anything unset) and apply them to audio_cfg. Returns the
    values in the widget order (== MODEL_OPTION_KEYS order)."""
    mid = _bare_model_id(model_label)
    _set_setting("model", mid)
    opts = _model_opts_for(mid)
    for k in MODEL_OPTION_KEYS:
        _set_setting(k, opts[k])
    shared.logger.info("audio_cpp: loaded per-model synthesis options for %r"
                       % (mid,))
    return [opts[k] for k in MODEL_OPTION_KEYS]


def _persist():
    """Save button: write global settings + every known per-model synthesis
    set to config.json. The currently selected model's set is taken from the
    live in-memory values."""
    global _model_options
    mid = _bare_model_id(audio_cfg.get("model"))
    if mid:
        opts = dict(_model_options.get(mid, {}))
        for k in MODEL_OPTION_KEYS:
            if k in audio_cfg:
                opts[k] = _coerce(k, audio_cfg[k])
        _model_options[mid] = opts
    data = {
        "global": {k: _coerce(k, audio_cfg[k]) for k in _DEFAULT_GLOBAL},
        "models": {m: {k: v for k, v in (o or {}).items()
                        if k in MODEL_OPTION_KEYS}
                   for m, o in _model_options.items()},
    }
    tmp = _SETTINGS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, _SETTINGS_FILE)
    shared.logger.info(
        "audio_cpp: config saved to %s (global + %d per-model option sets, "
        "active model %r)" % (_SETTINGS_FILE, len(data["models"]), mid))
    return "Saved to config.json (global + %d per-model option sets, " \
           "active model %r). Voicing %s." \
           % (len(data["models"]), mid,
              "ON" if audio_cfg.get("enabled") else "off")


def custom_css():
    return """
    .audio-cpp-player { margin: 0.5em 0; width: 100%; max-width: 480px; }

    /* Voice overlay: inline audio + delete, anchored under an assistant
       message's .text. Rendered idempotently by player.js. */
    .audio-cpp-voice {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0.5em 0 0;
        width: 100%;
    }
    .audio-cpp-voice audio {
        flex: 1;
        min-width: 0;
        max-width: 480px;
        height: 36px;
    }
    .audio-cpp-voice .aac-vlabel {
        font-size: 0.78em;
        opacity: 0.7;
        white-space: nowrap;
    }
    .audio-cpp-voice .aac-del {
        background: none;
        border: 1px solid rgba(255, 90, 90, 0.5);
        color: #ff8a8a;
        border-radius: 6px;
        font-size: 0.75em;
        padding: 2px 8px;
        cursor: pointer;
    }
    .audio-cpp-voice .aac-del:hover { background: rgba(255, 90, 90, 0.15); }
    """


def custom_js():
    js = "window.AUDIOCPP_ENABLED = %s;\n" % (
        "true" if audio_cfg.get("enabled") else "false")
    return js + _read("player.js")


def _read(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception:
        return ""


# Load persisted config at startup (module import == startup; ui() runs
# afterwards and builds every widget from the already-loaded values).
_load_config()


# expose for the player.js / relay
def get_audio_cfg():
    return dict(audio_cfg)