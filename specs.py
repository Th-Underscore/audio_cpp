"""Per-family request-option allow-lists for audio.cpp, loaded from specs.

audio.cpp validates every key in a speech request's `options` against the
model's model-contract `request_option_keys` (spec_backed_model.h:59) and
rejects the WHOLE request with "unknown <model> request option: <key>".
There is no HTTP endpoint exposing the spec, so the extension filters
options client-side before sending.

The allow-lists come from `model_specs/*.json` — exact copies of the
audio.cpp in-repo specs (Apache-2.0, see model_specs/LICENSE), which the
user may edit or replace directly to match their server build.

`detect_family()` maps a model id to a family using:
  1. explicit "family" fields in the local spec files (authoritative for
     ids that appear in a spec's options);
  2. filename heuristics (the GGUF filename always starts with the family
     name, e.g. breeze-tts-2-q8_0.gguf).

If no family is detected, filter_options() passes options through
UNFILTERED — the server remains the source of truth and a bad option just
fails that one request with a clear message.
"""
import json
import logging
import os
import re

SPEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_specs")
_dbg_fn = None


def set_logger(fn):
    global _dbg_fn
    _dbg_fn = fn


def _dbg(msg):
    if _dbg_fn:
        _dbg_fn(msg)
    else:
        logging.getLogger("textgen").info("audio_cpp: %s", msg)


def _load_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _options_of(spec):
    """The spec's request options, where they live in the JSON.

    Known shapes: a top-level "options" list, or a "request" object whose
    "options" is a list. Each entry is a dict with a "name" key, or a
    bare string. Tolerant: anything unreadable yields nothing.
    """
    opts = spec.get("options")
    if isinstance(opts, dict):
        req = opts.get("request")
        if isinstance(req, list):
            opts = req
        else:
            opts = None
    if not isinstance(opts, list):
        req = spec.get("request")
        if isinstance(req, dict) and isinstance(req.get("options"), list):
            opts = req["options"]
        elif isinstance(req, list):
            opts = req
    if not isinstance(opts, list):
        return []
    out = []
    for o in opts:
        if isinstance(o, dict):
            n = o.get("name")
            if isinstance(n, str) and n:
                out.append(n)
        elif isinstance(o, str) and o:
            out.append(o)
    return out


def _build():
    """Scan SPEC_DIR once -> ({family: frozenset(allowed)}, {id: family})."""
    allowed = {}
    id_map = {}
    if not os.path.isdir(SPEC_DIR):
        return allowed, id_map
    for fn in sorted(os.listdir(SPEC_DIR)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(SPEC_DIR, fn)
        try:
            spec = _load_spec(path)
        except Exception as e:
            _dbg("specs: failed to parse %s: %s" % (fn, e))
            continue
        if not isinstance(spec, dict):
            continue
        fam = None
        if isinstance(spec.get("family"), str) and spec["family"]:
            fam = spec["family"]
        else:
            fam = os.path.splitext(fn)[0]
        if not fam:
            continue
        opts = _options_of(spec)
        if not opts:
            _dbg("specs: %s has no request options (skipped)" % fn)
            continue
        if fam in allowed:
            _dbg("specs: duplicate family %s (%s) — first definition kept"
                 % (fam, fn))
            continue
        allowed[fam] = frozenset(opts)
        for o in opts:
            # a bare option name that looks like a model id is unusual;
            # only record explicit id fields if the spec has them
            pass
        for m in spec.get("models") or []:
            if isinstance(m, dict) and isinstance(m.get("id"), str):
                id_map.setdefault(m["id"], fam)
            elif isinstance(m, str):
                id_map.setdefault(m, fam)
    return allowed, id_map


_ALLOWED, _ID_MAP = _build()

# Filename heuristics: GGUF filenames start with the family's model name.
# Order matters: longer / more specific patterns first.
_PATTERNS = (
    (re.compile(r"^breeze[-_]?tts", re.I), "breeze_tts"),
    (re.compile(r"^vibevoice[-_]?asr[-_]?streaming", re.I), "vibevoice_asr_streaming"),
    (re.compile(r"^vibevoice[-_]?asr", re.I), "vibevoice_asr"),
    (re.compile(r"^vibevoice", re.I), "vibevoice"),
    (re.compile(r"^vibeasr", re.I), "vibeasr"),
    (re.compile(r"^kokoro[-_]?tts", re.I), "kokoro_tts"),
    (re.compile(r"^mira[-_]?tts", re.I), "mira_tts"),
    (re.compile(r"^outetts", re.I), "outetts"),
    (re.compile(r"^sopro[-_]?tts", re.I), "sopro_tts"),
    (re.compile(r"^soprano[-_]?tts", re.I), "soprano_tts"),
    (re.compile(r"^qwen3[-_]?tts", re.I), "qwen3_tts"),
    (re.compile(r"^f5[-_]?tts", re.I), "f5_tts"),
    (re.compile(r"^cosyvoice3?", re.I), "cosyvoice3"),
    (re.compile(r"^dots[-_]?tts", re.I), "dots_tts"),
    (re.compile(r"^fish[-_]?audio", re.I), "fish_audio"),
    (re.compile(r"^echo[-_]?tts", re.I), "echo_tts"),
    (re.compile(r"^glmtts|glm[-_]?tts", re.I), "glm_tts"),
    (re.compile(r"^neuttts?", re.I), "neutts"),
    (re.compile(r"^chatterbox[-_]?turbo", re.I), "chatterbox_turbo"),
    (re.compile(r"^chatterbox", re.I), "chatterbox"),
    (re.compile(r"^confucius4", re.I), "confucius4_tts"),
    (re.compile(r"^index[-_]?tts2?", re.I), "index_tts2"),
    (re.compile(r"^iran?odori[-_]?tts", re.I), "irodori_tts"),
    (re.compile(r"^magpie[-_]?tts", re.I), "magpie_tts"),
    (re.compile(r"^moss[-_]?tts[-_]?nano", re.I), "moss_tts_nano"),
    (re.compile(r"^moss[-_]?tts", re.I), "moss_tts_local"),
    (re.compile(r"^moss[-_]?voicegen", re.I), "moss_voicegen"),
    (re.compile(r"^minimax[-_]?h3", re.I), "minimax_h3"),
    (re.compile(r"^minimax[-_]?music3", re.I), "minimax_music3"),
    (re.compile(r"^sanotts", re.I), "sanotts"),
    (re.compile(r"^supertonic", re.I), "supertonic"),
    (re.compile(r"^vevo2?", re.I), "vevo2"),
    (re.compile(r"^vietneu[-_]?tts", re.I), "vietneu_tts"),
    (re.compile(r"^voxcpm2", re.I), "voxcpm2"),
    (re.compile(r"^voxcpm1", re.I), "voxcpm1"),
    (re.compile(r"^pocket[-_]?tts", re.I), "pocket_tts"),
    (re.compile(r"^dramabox", re.I), "dramabox"),
    (re.compile(r"^personaplex", re.I), "personaplex"),
    (re.compile(r"^parakeet[-_]?tdt", re.I), "parakeet_tdt"),
    (re.compile(r"^moonshine[-_]?asr", re.I), "moonshine_asr"),
    (re.compile(r"^omnivoice", re.I), "omnivoice"),
    (re.compile(r"^muscriptor", re.I), "muscriptor"),
    (re.compile(r"^heartmula", re.I), "heartmula"),
    (re.compile(r"^higgs[-_]?audio[-_]?stt", re.I), "higgs_audio_stt"),
    (re.compile(r"^higgs[-_]?audio[-_]?tts", re.I), "higgs_audio_tts"),
    (re.compile(r"^hvske[-_]?asr|hifiske[-_]?asr", re.I), "hviske_asr"),
    (re.compile(r"^inflect[-_]?v2", re.I), "inflect_v2"),
    (re.compile(r"^kroko[-_]?asr", re.I), "kroko_asr"),
    (re.compile(r"^sense[-_]?asr", re.I), "sense_asr"),
    (re.compile(r"^qwen3[-_]?asr", re.I), "qwen3_asr"),
    (re.compile(r"^qwen3[-_]?forced[-_]?aligner", re.I), "qwen3_forced_aligner"),
    (re.compile(r"^fireredtts3", re.I), "fireredtts3"),
    (re.compile(r"^firered[-_]?audio", re.I), "firered_audio"),
    (re.compile(r"^fun[-_]?asr[-_]?nano", re.I), "fun_asr_nano"),
    (re.compile(r"^granted?5asr|granite5asr", re.I), "granite5asr"),
    (re.compile(r"^nemotron[-_]?asr", re.I), "nemotron_asr"),
    (re.compile(r"^citrinet[-_]?asr", re.I), "citrinet_asr"),
    (re.compile(r"^audio8[-_]?asr", re.I), "audio8_asr"),
    (re.compile(r"^audio8[-_]?tts", re.I), "audio8_tts"),
    (re.compile(r"^sortformer[-_]?diar[-_]?v2", re.I), "sortformer_diar_v2"),
    (re.compile(r"^sortformer[-_]?diar", re.I), "sortformer_diar"),
    (re.compile(r"^mms[-_]?forced[-_]?aligner", re.I), "mms_forced_aligner"),
    (re.compile(r"^stable[-_]?audio", re.I), "stable_audio"),
    (re.compile(r"^controlfoley", re.I), "controlfoley"),
    (re.compile(r"^rvc", re.I), "rvc"),
    (re.compile(r"^seed[-_]?vc", re.I), "seed_vc"),
    (re.compile(r"^meanvc2", re.I), "meanvc2"),
    (re.compile(r"^miotts", re.I), "miotts"),
    (re.compile(r"^miocodec", re.I), "miocodec"),
    (re.compile(r"^midashenglm", re.I), "midashenglm_gen"),
    (re.compile(r"^bs[-_]?roformer", re.I), "bs_roformer"),
    (re.compile(r"^mel[-_]?band[-_]?roformer", re.I), "mel_band_roformer"),
    (re.compile(r"^htdemucs", re.I), "htdemucs"),
    (re.compile(r"^audiosr", re.I), "audiosr"),
    (re.compile(r"^ace[-_]?step", re.I), "ace_step"),
    (re.compile(r"^voxtral[-_]?realtime", re.I), "voxtral_realtime"),
)


def detect_family(model_id):
    """Family name for a model id, or None if unknown."""
    mid = (model_id or "").strip()
    if not mid:
        return None
    if mid in _ID_MAP:
        return _ID_MAP[mid]
    stem = re.sub(r"\.(gguf|bin|safetensors)$", "", mid, flags=re.I)
    for pat, fam in _PATTERNS:
        if pat.match(stem):
            return fam
    return None


def filter_options(model_id, options):
    """Drop request options the model's spec allow-list rejects.

    Returns (filtered_options, dropped_keys, family). Unknown families pass
    options through unfiltered.
    """
    fam = detect_family(model_id)
    if not fam:
        return options, [], None
    allowed = _ALLOWED.get(fam)
    if not allowed:
        return options, [], fam
    kept = {k: v for k, v in options.items() if k in allowed}
    dropped = sorted(k for k in options if k not in allowed)
    return kept, dropped, fam