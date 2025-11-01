# higgs_audio-non-vllm-version.py  (Windows client — remote-only + SFX)
import gradio as gr
import requests
import json
import os
from datetime import datetime
from pathlib import Path

BASE_URL = os.environ.get("HIGGS_BASE_URL", "http://10.6.1.15:8010").rstrip("/")

DEFAULT_TEMP = 1.0
DEFAULT_TOPP = 0.95
DEFAULT_TOPK = 50
DEFAULT_RAS_WIN = 7

CLIENT_CACHE = Path(__file__).resolve().parent / "client_outputs"
CLIENT_CACHE.mkdir(parents=True, exist_ok=True)

# -------- Server discovery --------
def fetch_server_voices(timeout_s=15):
    try:
        r = requests.get(f"{BASE_URL}/voices", timeout=timeout_s)
        if not r.ok:
            return [], f"/voices HTTP {r.status_code}: {r.text[:200]}"
        data = r.json()
        # Build a concise debug summary if available
        dbg = data.get("debug") or {}
        dbg_hint = ""
        try:
            ra = dbg.get("ref_analysis") or {}
            voice_hint = ra.get("voice_hint")
            f0 = ra.get("f0_hz")
            cond = dbg.get("conditioned")
            tokens = dbg.get("ref_tokens") or {}
            tokc = tokens.get("count")
            refb = dbg.get("ref_basename")
            dbg_hint = f" | cond={cond} f0={f0}Hz hint={voice_hint} tokc={tokc} ref={refb}"
        except Exception:
            pass
        if not data.get("ok"):
            return [], "Server returned ok=false"
        voices = data.get("voices", [])
        return sorted(voices) if isinstance(voices, list) else [], "ok"
    except Exception as e:
        return [], f"Exception fetching /voices: {e}"

def fetch_server_sfx(timeout_s=15):
    try:
        r = requests.get(f"{BASE_URL}/sfx", timeout=timeout_s)
        if not r.ok:
            return {"ok": False, "error": f"/sfx HTTP {r.status_code}: {r.text[:200]}"}
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def build_voice_choices(voices):
    return ["(none)"] + voices + ["(custom wav path)"]

def refresh_voices_server():
    vs, info = fetch_server_voices()
    return gr.update(choices=build_voice_choices(vs), value="(none)"), f"[SERVER] {BASE_URL}/voices  result={info}  found={len(vs)}"

def refresh_sfx():
    data = fetch_server_sfx()
    if not data.get("ok", False):
        return gr.update(choices=["(none)"], value="(none)"), f"[SFX] error: {data.get('error','unknown')}"
    bgm = data.get("bgm", [])
    choices = ["(none)"] + bgm
    status = f"[SFX] laughter={data.get('laughter')} applause={data.get('applause')} bgm_tracks={len(bgm)}"
    return gr.update(choices=choices, value="(none)"), status

# -------- Helpers --------
def safe_stub(stub: str) -> str:
    s = (stub or "").strip().strip("/\\")
    return s if s else datetime.now().strftime("higgs_%Y%m%d_%H%M%S")

def _download_audio(audio_url: str, stub: str) -> str:
    if not stub.lower().endswith(".wav"):
        stub = f"{stub}.wav"
    local_path = CLIENT_CACHE / stub
    with requests.get(audio_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    return str(local_path)

# -------- Generate --------
def generate_audio(text, voice_choice, custom_wav_path,
                   temperature, top_p, top_k, ras_win_len, filename_stub,
                   sfx_laughter, sfx_applause, bgm_name, sfx_gain_db, bgm_gain_db, sfx_offset_ms, bgm_offset_ms):
    stub = safe_stub(filename_stub)

    payload = {
        "transcript": text,
        "out_path": stub,  # server will force into outputs/ and append .wav
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "ras_win_len": int(ras_win_len),
        "ras_win_max_num_repeat": 2,
        "generation_chunk_buffer_size": None,
        "seed": 42,  # keep deterministic unless you want creative mode
        # ---- SFX ----
        "sfx_laughter": bool(sfx_laughter),
        "sfx_applause": bool(sfx_applause),
        "bgm_name": None if (bgm_name in (None, "", "(none)")) else bgm_name,
        "sfx_gain_db": float(sfx_gain_db),
        "bgm_gain_db": float(bgm_gain_db),
        "sfx_offset_ms": int(sfx_offset_ms),
        "bgm_offset_ms": int(bgm_offset_ms),
    }

    if voice_choice == "(custom wav path)":
        if not custom_wav_path:
            return None, "Custom WAV path required (path on the SERVER).", json.dumps(payload, indent=2)
        payload["ref_audio_wav_path"] = custom_wav_path.replace("\\", "/")
    elif voice_choice not in ("(none)", "(custom wav path)"):
        payload["ref_audio"] = voice_choice

    prompt_str = json.dumps(payload, indent=2)

    try:
        r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=600)
        if not r.ok:
            return None, f"HTTP {r.status_code}: {r.text}", prompt_str
        data = r.json()
        # Build a concise debug summary if available
        dbg = data.get("debug") or {}
        dbg_hint = ""
        try:
            ra = dbg.get("ref_analysis") or {}
            voice_hint = ra.get("voice_hint")
            f0 = ra.get("f0_hz")
            cond = dbg.get("conditioned")
            tokens = dbg.get("ref_tokens") or {}
            tokc = tokens.get("count")
            refb = dbg.get("ref_basename")
            dbg_hint = f" | cond={cond} f0={f0}Hz hint={voice_hint} tokc={tokc} ref={refb}"
        except Exception:
            pass
        if not data.get("ok"):
            return None, f"Server error: {data.get('error')}\n{data.get('traceback','')}", prompt_str

        rel = data.get("relative_url")
        if not rel:
            basename = os.path.basename(data.get("out_path", f"{stub}.wav"))
            rel = f"/outputs/{basename}"
        audio_url = f"{BASE_URL}{rel}"

        local_audio = _download_audio(audio_url, stub)
        return local_audio, f"OK  |  SFX: {data.get('sfx_applied')}{dbg_hint}", prompt_str

    except Exception as e:
        return None, f"Connection error: {e}", prompt_str

# -------- UI --------
with gr.Blocks(title="Higgs Audio 2 — Remote Client") as demo:
    gr.Markdown(
        "## 🎤 Higgs Audio 2 — Remote Client with SFX\n"
        f"**Server:** `{BASE_URL}`  •  Voices via /voices  •  SFX via /sfx  •  Audio cached locally."
    )

    with gr.Row():
        text_input = gr.Textbox(label="Text to Speak", lines=3, placeholder="Type something to speak...")

    with gr.Row():
        voice_dropdown = gr.Dropdown(choices=["(none)"], value="(none)", label="Reference Voice")
        custom_wav     = gr.Textbox(
            label="Custom WAV Path (on SERVER; only if selected)",
            placeholder="/home/mintdude/Github/sparky/higgs/examples/voice_prompts/belinda.wav"
        )

    with gr.Accordion("Voice & Style", open=True):
        with gr.Row():
            temperature = gr.Slider(0.1, 1.5, value=DEFAULT_TEMP, step=0.05, label="Temperature")
            top_p       = gr.Slider(0.1, 1.0, value=DEFAULT_TOPP, step=0.01, label="Top-p")
            top_k       = gr.Slider(1, 200, value=DEFAULT_TOPK, step=1, label="Top-k")
            ras_win_len = gr.Slider(0, 20, value=DEFAULT_RAS_WIN, step=1, label="RAS win len (0=off)")

    with gr.Accordion("SFX (Laughter, Applause, Background Music)", open=True):
        with gr.Row():
            sfx_laughter = gr.Checkbox(label="Laughter", value=False)
            sfx_applause = gr.Checkbox(label="Applause", value=False)
            bgm_dropdown = gr.Dropdown(choices=["(none)"], value="(none)", label="Background Music")
        with gr.Row():
            sfx_gain_db   = gr.Slider(-36, 6, value=-8, step=1, label="SFX Gain (dB)")
            bgm_gain_db   = gr.Slider(-36, 6, value=-16, step=1, label="BGM Gain (dB)")
        with gr.Row():
            sfx_offset_ms = gr.Slider(0, 30000, value=0, step=100, label="SFX Offset (ms)")
            bgm_offset_ms = gr.Slider(0, 30000, value=0, step=100, label="BGM Offset (ms)")
        with gr.Row():
            refresh_sfx_btn = gr.Button("Refresh SFX")

    with gr.Row():
        filename_stub = gr.Textbox(label="Output filename (no .wav)", placeholder="higgs_test")

    with gr.Row():
        generate_btn = gr.Button("Generate & Play", variant="primary")
        refresh_btn  = gr.Button("Refresh Voices")

    with gr.Row():
        audio_output   = gr.Audio(label="Generated Audio", type="filepath")
    with gr.Row():
        status_msg     = gr.Textbox(label="Status", interactive=False, lines=18)
        prompt_display = gr.Textbox(label="JSON Sent to Server", interactive=False, lines=18)

    generate_btn.click(
        fn=generate_audio,
        inputs=[text_input, voice_dropdown, custom_wav,
                temperature, top_p, top_k, ras_win_len, filename_stub,
                sfx_laughter, sfx_applause, bgm_dropdown, sfx_gain_db, bgm_gain_db, sfx_offset_ms, bgm_offset_ms],
        outputs=[audio_output, status_msg, prompt_display]
    )

    refresh_btn.click(fn=refresh_voices_server, inputs=[], outputs=[voice_dropdown, status_msg])
    refresh_sfx_btn.click(fn=refresh_sfx, inputs=[], outputs=[bgm_dropdown, status_msg])

    demo.load(fn=refresh_voices_server, inputs=[], outputs=[voice_dropdown, status_msg])
    demo.load(fn=refresh_sfx, inputs=[], outputs=[bgm_dropdown, status_msg])

demo.launch(server_name="0.0.0.0", server_port=7860)
