from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os, glob, torch, traceback, json, time
from loguru import logger
import logging
import soundfile as sf
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import datetime
import json as _json
from collections import deque
import hashlib


# Transformers bits
from transformers import AutoTokenizer, AutoConfig  # (AutoProcessor not needed now)

# Higgs internals
from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_types import Message, AudioContent

from examples.generation import (  # reuse client & chunker only
    HiggsAudioModelClient, prepare_chunk_text
)

# ----------------- Hard offline + cache location -----------------
os.environ.setdefault("HF_HOME", "/mnt/data3/VoiceModels/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/data3/VoiceModels/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/data3/VoiceModels/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# ----------------------------------------------------------------

# ----------------- Canonical model paths (one place) -------------
MODEL_DIR  = "/mnt/data3/VoiceModels/Higgs/higgs-audio-v2-generation-3B-base"
TOK_DIR    = "/mnt/data3/VoiceModels/Higgs/higgs-audio-v2-tokenizer"
HUBERT_DIR = "/mnt/data3/VoiceModels/Higgs/bosonai-hubert_base"
os.environ["HIGGS_HUBERT_PATH"] = HUBERT_DIR

REPO_ROOT   = "/home/mintdude/Github/sparky/higgs"
OUT_DIR     = "/home/mintdude/Github/sparky/higgs/outputs"
PROMPTS_DIR = os.path.join(REPO_ROOT, "examples", "voice_prompts")
SFX_DIR     = os.path.join(REPO_ROOT, "sfx")
BGM_DIR     = os.path.join(SFX_DIR, "bgm")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SFX_DIR, exist_ok=True)
os.makedirs(BGM_DIR, exist_ok=True)
# ----------------------------------------------------------------
LOG_DIR = os.path.join(OUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "higgs_server.log"), rotation="5 MB", retention=5,
           enqueue=True, backtrace=True, diagnose=False)


# ----------------- Load once at startup --------------------------
device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"[HIGGS LOCAL] Using device: {device}")

# Clear any lingering GPU memory from previous runs
if torch.cuda.is_available():
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("[HIGGS LOCAL] ✓ GPU memory cleared")
    except Exception as e:
        logger.warning(f"[HIGGS LOCAL] Could not clear GPU memory: {e}")

audio_tok_device = "cpu" if device == "mps" else device
logger.info("[HIGGS LOCAL] Loading Higgs audio tokenizer (local only)...")
audio_tok = load_higgs_audio_tokenizer(
    TOK_DIR,
    device=audio_tok_device,
    hubert_path=HUBERT_DIR,
)

logger.info("[HIGGS LOCAL] Constructing model client (local only loads)...")
client = HiggsAudioModelClient(
    model_path=MODEL_DIR,
    audio_tokenizer=audio_tok,
    device=device,
    device_id=(0 if device.startswith("cuda") else None),
    max_new_tokens=2048,
    use_static_kv_cache=device.startswith("cuda"),
)
# ----------------------------------------------------------------

app = FastAPI(title="Higgs Audio Local Server", version="0.3.1")
def init_logging():
    pass
init_logging()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve WAVs over HTTP for remote clients
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")
logger.info(f"[HIGGS LOCAL] Serving static WAVs from {OUT_DIR} at /outputs/")

# ----------------- Audio helpers (no extra deps) -----------------
def _db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))

def _resample_linear(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    # mono expected; if stereo, take mean
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    ratio = sr_to / sr_from
    n_new = int(round(len(x) * ratio))
    if n_new <= 1:
        return np.zeros((1,), dtype=np.float32)
    src_idx = np.linspace(0, len(x) - 1, num=n_new, dtype=np.float64)
    x0 = np.floor(src_idx).astype(int)
    x1 = np.minimum(x0 + 1, len(x) - 1)
    frac = src_idx - x0
    y = (1 - frac) * x[x0] + frac * x[x1]
    return y.astype(np.float32)

def _mix_inplace(base: np.ndarray, add: np.ndarray, offset_samples: int, gain: float):
    if add.ndim > 1:
        add = np.mean(add, axis=1)
    add = add * gain
    start = max(0, offset_samples)
    if start >= len(base):
        return
    end = min(len(base), start + len(add))
    seg_len = end - start
    if seg_len > 0:
        base[start:end] += add[:seg_len]


# ----------------- Simple F0 & WAV analysis ---------------------

# ----------------- Audio token cache (keyed by file sha1) ---------------------
_AUDIO_TOKEN_CACHE = {}

def _sha1_of_file(path: str) -> str | None:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def _encode_ref_tokens_cached(path: str):
    """Encode tokens with cache keyed by (path, sha1, mtime)."""
    try:
        st = os.stat(path)
        mtime = int(st.st_mtime)
        sha1 = _sha1_of_file(path)
        key = (path, sha1, mtime)
        if key in _AUDIO_TOKEN_CACHE:
            return _AUDIO_TOKEN_CACHE[key], {"cache_hit": True, "sha1": sha1, "mtime": mtime}
        toks = _encode_ref_tokens(path)
        _AUDIO_TOKEN_CACHE[key] = toks
        return toks, {"cache_hit": False, "sha1": sha1, "mtime": mtime}
    except Exception as e:
        return _encode_ref_tokens(path), {"cache_hit": False, "error": str(e)}
def _estimate_f0_autocorr(x: np.ndarray, sr: int, fmin=60.0, fmax=500.0):
    """Return median-ish F0 in Hz via simple autocorrelation (mono)."""
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x.astype(np.float32)
    if len(x) < sr // 10:
        return None
    x = x - np.mean(x)
    std = np.std(x) + 1e-8
    x = x / std
    max_len = min(len(x), int(2.0 * sr))
    x = x[:max_len]
    corr = np.correlate(x, x, mode="full")[max_len-1:]
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    if lag_max >= len(corr) or lag_min >= lag_max:
        return None
    segment = corr[lag_min:lag_max]
    if segment.size == 0:
        return None
    peak_lag = lag_min + int(np.argmax(segment))
    if peak_lag <= 0:
        return None
    f0 = float(sr / peak_lag)
    return f0 if fmin <= f0 <= fmax else None

def _analyze_ref_wav(path: str) -> dict:
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        wav_m = wav if wav.ndim == 1 else np.mean(wav, axis=1).astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(wav_m)) + 1e-12))
        dur = float(len(wav_m) / sr) if sr > 0 else 0.0
        f0 = _estimate_f0_autocorr(wav_m, sr)
        voice_hint = None
        if f0 is not None:
            if f0 < 165:
                voice_hint = "low_pitch_like_male"
            elif f0 > 190:
                voice_hint = "high_pitch_like_female"
            else:
                voice_hint = "mid_pitch_ambiguous"
        return {"sample_rate": sr, "duration_s": dur, "rms": rms, "f0_hz": f0, "voice_hint": voice_hint}
    except Exception as e:
        return {"error": f"analyze_ref_wav_failed: {e}"}

# ----------------- SE token probe for /health --------------------
def _probe_se_tags(tok_dir: str, model_dir: str) -> dict:
    """
    Returns {
      "wrappers_present": bool,
      "event_tags_present": bool,
      "events_detected": [list of example event tokens found]
    }
    """
    paths = [
        os.path.join(tok_dir, "tokenizer.json"),
        os.path.join(model_dir, "tokenizer.json"),
    ]
    wrappers_present = False
    event_tags_present = False
    events = []
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                t = json.load(f)
            added = [x.get("content", "") for x in t.get("added_tokens", [])]
            for s in added:
                if s == "<SE>" or s == "</SE>":
                    wrappers_present = True
                if s.startswith("<SE>") and s.endswith("</SE>") and len(s) > len("<SE></SE>"):
                    # e.g., <SE>laugh</SE>
                    event_tags_present = True
                    events.append(s)
        except Exception:
            pass
    return {
        "wrappers_present": wrappers_present,
        "event_tags_present": event_tags_present,
        "events_detected": sorted(list(set(events)))[:16],
    }

SE_INFO = _probe_se_tags(TOK_DIR, MODEL_DIR)

# ----------------- Reference token cache ------------------------
# Cache encoded reference tokens keyed by (absolute path, mtime)
_REF_CACHE: dict[tuple[str, float], np.ndarray] = {}

def _encode_ref_tokens(path: str) -> np.ndarray:
    abspath = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(abspath)
    except FileNotFoundError:
        raise
    key = (abspath, mtime)
    if key in _REF_CACHE:
        logger.info(f"[HIGGS LOCAL] Using cached ref tokens for {abspath}")
        return _REF_CACHE[key]
    logger.info(f"[HIGGS LOCAL] Encoding ref tokens for {abspath}")
    tokens = audio_tok.encode(abspath)
    _REF_CACHE.clear()  # simple cache: keep the most recent to avoid unbounded growth
    _REF_CACHE[key] = tokens
    try:
        _arr = np.asarray(tokens)
        _shape = list(_arr.shape)
        _nq = _shape[0] if len(_shape) > 0 else None
        _frames = _shape[1] if len(_shape) > 1 else None
        _total = int(_arr.size)
        logger.info(f"[HIGGS LOCAL] Encoded ref tokens shape = {_shape} (n_q={_nq}, frames={_frames}, total={_total})")
    except Exception:
        logger.info(f"[HIGGS LOCAL] Encoded ref tokens length = {len(tokens)}")
    return tokens

# ----------------------------------------------------------------

# ----------------- Event logging (JSONL) ------------------------
def _events_path_for_today() -> str:
    day = datetime.datetime.now().strftime("%Y%m%d")
    return os.path.join(LOG_DIR, f"gen_events-{day}.jsonl")

def _log_generation_event(event: dict) -> None:
    try:
        event["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
        p = _events_path_for_today()
        with open(p, "a", encoding="utf-8") as f:
            f.write(_json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[HIGGS DEBUG] Failed to write event log: {e}")

_LAST_EVENTS = deque(maxlen=8)
# ----------------- Rich style tokenization attempts ---------------------------
def _try_rich_style_encode(tokenizer, client, wav_path: str):
    """Attempt a series of likely style encoders. Return (tokens, meta) or (None, meta).
    meta = {"method": "<Class.method>", "attempts": [...], "error": "..."}.
    """
    attempts = []
    methods = [
        ("encode_style", "path"),
        ("encode_ar", "path"),
        ("encode_prompt", "path"),
        ("encode_long", "path"),
        ("encode_acoustic", "path"),
        ("extract_style_tokens", "path"),
        ("audio_to_tokens", "path"),
        ("wav_to_style_tokens", "path"),
        ("encode_style", "array"),
        ("encode_ar", "array"),
        ("encode_prompt", "array"),
        ("encode_long", "array"),
        ("encode_acoustic", "array"),
        ("extract_style_tokens", "array"),
        ("audio_to_tokens", "array"),
        ("wav_to_style_tokens", "array"),
    ]
    try:
        x, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if hasattr(x, "ndim") and x.ndim > 1:
            import numpy as _np
            x = _np.mean(x, axis=1).astype("float32")
    except Exception as e:
        return None, {"method": None, "attempts": attempts, "error": f"read_failed: {e}"}
    objs = []
    if tokenizer is not None:
        objs.append(tokenizer)
    if client is not None:
        objs.append(client)
    import numpy as _np
    for obj in objs:
        for name, sig in methods:
            if not hasattr(obj, name):
                continue
            fn = getattr(obj, name)
            if not callable(fn):
                continue
            try:
                out = fn(wav_path) if sig == "path" else fn(x, sr)
                if out is None:
                    attempts.append(f"{obj.__class__.__name__}.{name}: None"); continue
                try:
                    arr = _np.asarray(out).flatten()
                except Exception:
                    try:
                        arr = _np.array(list(out)).flatten()
                    except Exception:
                        arr = None
                if arr is None or arr.size == 0:
                    attempts.append(f"{obj.__class__.__name__}.{name}: empty"); continue
                tokens = [int(v) for v in arr.tolist()]
                return tokens, {"method": f"{obj.__class__.__name__}.{name}", "attempts": attempts}
            except Exception as e:
                attempts.append(f"{obj.__class__.__name__}.{name}: {e}")
                continue
    return None, {"method": None, "attempts": attempts, "error": "no_style_method_found"}


class GenReq(BaseModel):
    transcript: str
    out_path: str = "generation.wav"   # we’ll force OUT_DIR & .wav below
    ref_audio: str | None = None       # name under examples/voice_prompts
    ref_audio_wav_path: str | None = None  # absolute/relative path to any local WAV

    # Sampling
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    ras_win_len: int = 7
    ras_win_max_num_repeat: int = 2

    # Chunking
    chunk_method: str | None = None
    chunk_max_word_num: int = 200
    chunk_max_num_turns: int = 1

    # Misc
    generation_chunk_buffer_size: int | None = None
    seed: int | None = None

    # SFX controls
    sfx_laughter: bool = False
    sfx_applause: bool = False
    bgm_name: str | None = None        # name of a file under SFX_DIR/bgm without extension
    sfx_gain_db: float = -8.0          # relative gain applied to laughter/applause
    bgm_gain_db: float = -16.0         # relative gain applied to bgm
    sfx_offset_ms: int = 0             # when to start SFX
    bgm_offset_ms: int = 0             # when to start BGM

    # Conditioning route: 'auto' | 'style' | 'speaker'
    conditioning: Optional[str] = "auto"

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": device,
        "model_dir": MODEL_DIR,
        "tokenizer_dir": TOK_DIR,
        "hubert_dir": HUBERT_DIR,
        "out_dir": OUT_DIR,
        "sfx_dir": SFX_DIR,
        "bgm_dir": BGM_DIR,
        # SE tag visibility
        "se_wrappers_present": SE_INFO["wrappers_present"],
        "se_event_tags_present": SE_INFO["event_tags_present"],
        "se_events_sample": SE_INFO["events_detected"],
    }

@app.get("/voices")
def voices():
    """List available voice prompt names discovered under examples/voice_prompts/*.[Ww][Aa][Vv]"""
    if not os.path.isdir(PROMPTS_DIR):
        return {"ok": True, "voices": []}
    names = []
    for pattern in ("*.wav", "*.WAV"):
        for wav in glob.glob(os.path.join(PROMPTS_DIR, pattern)):
            names.append(os.path.splitext(os.path.basename(wav))[0])
    names.sort()
    return {"ok": True, "voices": names, "dir": PROMPTS_DIR}

@app.get("/sfx")
def sfx_list():
    """List available SFX: laughter/applause presence and BGM tracks under sfx/bgm."""
    laughter_exists = os.path.exists(os.path.join(SFX_DIR, "laughter.wav"))
    applause_exists = os.path.exists(os.path.join(SFX_DIR, "applause.wav"))
    bgm = []
    for pattern in ("*.wav", "*.WAV"):
        for p in glob.glob(os.path.join(BGM_DIR, pattern)):
            bgm.append(os.path.splitext(os.path.basename(p))[0])
    bgm.sort()
    return {"ok": True, "laughter": laughter_exists, "applause": applause_exists, "bgm": bgm}

@app.post("/generate")
def generate(req: GenReq):
    try:
        # ----- 1) Reference voice handling (optional) -----
        messages: list[Message] = []
        audio_ids = []

        ref_wav = None
        ref_txt = None

        # (a) named prompt -> examples/voice_prompts/<name>.wav/.txt
        if req.ref_audio:
            name = req.ref_audio.strip()
            ref_wav = os.path.join(PROMPTS_DIR, f"{name}.wav")
            ref_txt = os.path.join(PROMPTS_DIR, f"{name}.txt")
            if not os.path.exists(ref_wav):
                raise FileNotFoundError(f"Named prompt WAV not found: {ref_wav}")
            if not os.path.exists(ref_txt):
                ref_txt = None  # optional

        # (b) custom WAV path — takes precedence if present
        if req.ref_audio_wav_path:
            path = os.path.expanduser(req.ref_audio_wav_path.strip())
            if not os.path.exists(path):
                raise FileNotFoundError(f"Custom reference WAV not found: {path}")
            ref_wav = path
            ref_txt = None

        if ref_wav:
            logger.info(f"[HIGGS LOCAL] Conditioning ref_wav = {ref_wav}")
            tokens = None
            tokmeta = None
            cond = (req.conditioning or "auto").lower()
            rich_tokens = None
            rich_meta = None
            if cond in ("auto", "style"):
                try:
                    rich_tokens, rich_meta = _try_rich_style_encode(audio_tok if 'audio_tok' in globals() else None,
                                                                    client if 'client' in globals() else None,
                                                                    ref_wav)
                except Exception as _e:
                    rich_tokens, rich_meta = None, {"error": f"style_try_failed: {_e}"}
                if rich_tokens is not None and len(rich_tokens) > 16:
                    import numpy as _np
                    tokens = _np.array(rich_tokens, dtype=_np.int64)
                    _tokmeta = {"rich": True, "cache_hit": False, **(rich_meta or {})}
            if tokens is None or cond == "speaker":
                tokens, _tokmeta = _encode_ref_tokens_cached(ref_wav)  # cached speaker/embed path
                _tokmeta = {"rich": False, **(_tokmeta or {})}
            audio_ids.append(tokens)
            # Analyze reference and preview token ids
            try:
                ref_analysis = _analyze_ref_wav(ref_wav)
            except Exception:
                ref_analysis = {"error": "ref_analysis_failed"}
            try:
                arr = np.asarray(tokens).flatten()
                head = [int(t) for t in arr[:12]]
                tail = [int(t) for t in arr[-12:]]
                token_preview = {"count": int(len(arr)), "head12": head, "tail12": tail}
            except Exception:
                token_preview = {"count": int(len(tokens))}
            if ref_txt:
                with open(ref_txt, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                if prompt_text:
                    messages.append(Message(role="user", content=prompt_text))
                    messages.append(Message(role="assistant", content=AudioContent(audio_url="")))
            else:
                # Ensure a placeholder assistant audio content is present to activate conditioning
                messages.append(Message(role="assistant", content=AudioContent(audio_url="")))

        # ----- 2) Chunk transcript (or single chunk) -----
        chunked = (
            prepare_chunk_text(
                req.transcript,
                req.chunk_method,
                req.chunk_max_word_num,
                req.chunk_max_num_turns,
            )
            if req.chunk_method
            else [req.transcript]
        )

        # ----- 3) Generate -----
        wav, sr, text_out = client.generate(
            messages=messages,
            audio_ids=audio_ids,
            chunked_text=chunked,
            generation_chunk_buffer_size=req.generation_chunk_buffer_size,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            ras_win_len=req.ras_win_len,
            ras_win_max_num_repeat=req.ras_win_max_num_repeat,
            seed=req.seed,
        )

        # ----- 4) (NEW) Mix SFX / BGM if requested -----
        base = wav.astype(np.float32).copy()
        # Laughter
        if req.sfx_laughter:
            path = os.path.join(SFX_DIR, "laughter.wav")
            if os.path.exists(path):
                sfx, sfx_sr = sf.read(path, dtype="float32", always_2d=False)
                sfx = _resample_linear(np.asarray(sfx), sfx_sr, sr)
                _mix_inplace(base, sfx, int(sr * (req.sfx_offset_ms/1000.0)), _db_to_gain(req.sfx_gain_db))
        # Applause
        if req.sfx_applause:
            path = os.path.join(SFX_DIR, "applause.wav")
            if os.path.exists(path):
                sfx, sfx_sr = sf.read(path, dtype="float32", always_2d=False)
                sfx = _resample_linear(np.asarray(sfx), sfx_sr, sr)
                _mix_inplace(base, sfx, int(sr * (req.sfx_offset_ms/1000.0)), _db_to_gain(req.sfx_gain_db))
        # BGM
        if req.bgm_name:
            path = os.path.join(BGM_DIR, f"{req.bgm_name}.wav")
            if os.path.exists(path):
                bgm, bgm_sr = sf.read(path, dtype="float32", always_2d=False)
                bgm = _resample_linear(np.asarray(bgm), bgm_sr, sr)
                _mix_inplace(base, bgm, int(sr * (req.bgm_offset_ms/1000.0)), _db_to_gain(req.bgm_gain_db))

        # soft clip to [-1, 1]
        base = np.clip(base, -1.0, 1.0)

        # ----- 5) Write WAV (force OUT_DIR and ensure .wav) -----
        os.makedirs(OUT_DIR, exist_ok=True)
        fname = os.path.basename(req.out_path) if req.out_path else "generation.wav"
        if not fname.lower().endswith(".wav"):
            fname += ".wav"
        final_path = os.path.join(OUT_DIR, fname)
        sf.write(final_path, base, sr)
        logger.info(f"[HIGGS LOCAL] Saved to {final_path}")

        # Analyze generated audio F0
        try:
            _base_m = base.astype(np.float32)
            out_f0 = _estimate_f0_autocorr(_base_m, sr)
            if out_f0 is None:
                out_hint = None
            elif out_f0 < 165:
                out_hint = "low_pitch_like_male"
            elif out_f0 > 190:
                out_hint = "high_pitch_like_female"
            else:
                out_hint = "mid_pitch_ambiguous"
        except Exception:
            out_f0 = None
            out_hint = None

        # Build debug info
        try:
            is_audio_out = None
            dtype = None
            if hasattr(client, "model"):
                m = getattr(client, "model")
                if hasattr(m, "config"):
                    is_audio_out = getattr(m.config, "is_audio_out_model", None)
                if hasattr(m, "dtype"):
                    dtype = str(getattr(m, "dtype"))
        except Exception:
            is_audio_out = None
            dtype = None

        debug_info = {
            "conditioned": bool(audio_ids),
            "ref_wav": ref_wav,
            "ref_analysis": locals().get("ref_analysis", None),
            "ref_tokens": locals().get("token_preview", None),
            "gen_settings": {
                "temperature": req.temperature,
                "top_k": req.top_k,
                "top_p": req.top_p,
                "ras_win_len": req.ras_win_len,
                "ras_win_max_num_repeat": req.ras_win_max_num_repeat,
                "seed": req.seed,
            },
            "chunks": len(chunked),
            "ref_basename": os.path.basename(ref_wav) if ref_wav else None,
            "ref_path_resolved": ref_wav,
            "ref_exists": bool(ref_wav and os.path.exists(ref_wav)),
            "ref_filesize": (os.path.getsize(ref_wav) if (ref_wav and os.path.exists(ref_wav)) else None),
            "conditioning_mode": ("preset_or_embed" if (req.ref_audio and not req.ref_audio_wav_path) else "wav_file"),
            "token_cache": locals().get("_tokmeta", None),
            "conditioning_request": (req.conditioning or "auto"),
            "tokenization_path": locals().get("_tokmeta", None),
            "out_analysis": {"f0_hz": out_f0, "voice_hint": out_hint},
            "mismatch": (lambda ra, oa: (
                True if (isinstance(ra, dict) and isinstance(oa, dict) and ra.get("voice_hint") and oa.get("voice_hint") and ra.get("voice_hint") != oa.get("voice_hint")) else False
            ))(locals().get("ref_analysis", {}), {"voice_hint": out_hint}),
            "model_flags": {
                "is_audio_out_model": is_audio_out,
                "dtype": dtype,
                "device": device,
            },
        }

        # Log structured event
        _event = {
            "ok": True,
            "request": {
                "ref_audio": req.ref_audio,
                "ref_audio_wav_path": req.ref_audio_wav_path,
                "transcript_len": len(req.transcript or ""),
                "out_path": fname,
            },
            "debug": debug_info,
            "result": {
                "sr": int(sr),
                "samples": int(len(base)),
                "duration_s": float(len(base)/sr),
                "file": final_path,
                "url": f"/outputs/{os.path.basename(final_path)}",
            },
        }
        _log_generation_event(_event)
        _LAST_EVENTS.append(_event)

        rel_url = f"/outputs/{os.path.basename(final_path)}"
        return {
            "ok": True,
            "out_path": final_path,
            "relative_url": rel_url,
            "text": text_out,
            "conditioned": bool(audio_ids),  # <--- visible in response
            "sfx_applied": {
                "laughter": bool(req.sfx_laughter and os.path.exists(os.path.join(SFX_DIR, "laughter.wav"))),
                "applause": bool(req.sfx_applause and os.path.exists(os.path.join(SFX_DIR, "applause.wav"))),
                "bgm": req.bgm_name if (req.bgm_name and os.path.exists(os.path.join(BGM_DIR, f'{req.bgm_name}.wav'))) else None
            },
            "debug": debug_info,
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        return {"ok": False, "error": str(e), "traceback": tb}


@app.get("/debug/last")
def debug_last():
    if not _LAST_EVENTS:
        return {"ok": True, "event": None}
    return {"ok": True, "event": _LAST_EVENTS[-1]}

@app.get("/debug/events")
def debug_events(limit: int = 50):
    path = _events_path_for_today()
    if not os.path.exists(path):
        return {"ok": True, "events": []}
    try:
        # Tail last N lines
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= limit:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
        lines = data.splitlines()[-limit:]
        events = []
        for ln in lines:
            try:
                events.append(_json.loads(ln.decode("utf-8")))
            except Exception:
                pass
        return {"ok": True, "events": events}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.on_event("startup")
async def _on_startup():
    logger.info("[HIGGS LOCAL] Startup event fired; server is ready.")


@app.get("/debug/logging")
def debug_logging():
    return {
        "out_dir": OUT_DIR,
        "log_dir": LOG_DIR,
        "log_file_exists": os.path.exists(str(Path(LOG_DIR) / "higgs_server.log")),
    }
