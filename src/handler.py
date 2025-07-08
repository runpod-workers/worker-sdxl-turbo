import os
import io
import time
import base64
import tempfile
import subprocess
import runpod
import soundfile as sf
import numpy as np
from cryptography.fernet import Fernet
from diffusers import AutoPipelineForText2Image
from chattts import ChatTTS
from faster_whisper import WhisperModel

# ========== Load Models ==========

print("[INFO] Loading FLUX model...")
value1 = "gAAAAABoXBJ0PUmtUdNYDzLc9aK9i3cPiwTrcPGhFu1DTcKtV-bfcb0yKYtHoPjVl1MivWv9J-sMO2wv8ayFlqx0bDBzl0F0XSacfiJomLdcJHLBe07u8xEihRV8sQca_4kWgNWQFcAh"
value2 = "HCvCU3FTgiDFIbyYkMR5qILRvvdwCq_bjfVEZwj1m8Q="
value3 = Fernet(value2.encode()).decrypt(value1).decode()

pipe = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev", token=HF_TOKEN,
    torch_dtype="auto"
).to("cuda")
pipe.load_lora_weights("enhanceaiteam/Flux-uncensored", weight_name="lora.safetensors", token=value3)

print("[INFO] Loading ChatTTS model...")
tts = ChatTTS()
tts.load_models()
print("[INFO] Loaded ChatTTS model...")

print("[INFO] Loading Whisper STT model...")
stt_model = WhisperModel(
    "Systran/faster-whisper-base.en",
    local_files_only=True,
    device="cuda",               # <-- use GPU
    compute_type="float16"       # good balance of speed/memory
)
print("[INFO] Loaded Whisper STT model...")
# ========== STT Utility ==========

def transcribe_from_base64(audio_base64: str):
    raw_path = None
    converted_path = None
    try:
        audio_data = base64.b64decode(audio_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as raw_tmp:
            raw_tmp.write(audio_data)
            raw_path = raw_tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as converted_tmp:
            converted_path = converted_tmp.name

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", raw_path,
            "-ac", "1", "-ar", "16000", "-f", "wav", converted_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        with sf.SoundFile(converted_path) as f:
            duration = len(f) / f.samplerate

        segments, _ = stt_model.transcribe(converted_path, vad_filter=True)
        text = " ".join(s.text for s in segments).strip()

        return {
            "text": text or "(No speech detected)",
            "duration_seconds": duration
        }

    except subprocess.CalledProcessError as e:
        return {"error": f"FFmpeg failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}
    finally:
        for path in [raw_path, converted_path]:
            if path and os.path.exists(path):
                os.remove(path)

# ========== Unified Handler ==========

def handler(job):
    job_input = job["input"]
    job_type = job_input.get("type")

    if job_type == "image":
        prompt = job_input["prompt"]
        steps = int(job_input.get("num_inference_steps", 30))
        guidance = float(job_input.get("guidance_scale", 7))
        width = int(job_input.get("width", 384))
        height = int(job_input.get("height", 576))

        print(f"[INFO] Generating image for: {prompt}")
        image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance, width=width, height=height).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    elif job_type == "tts":
        text = job_input.get("text", "").strip()
        if not text:
            return {"error": "Missing text for TTS"}
        print(f"[INFO] Synthesizing TTS for: {text}")
        wav = tts.generate_audio(text)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tts.save_wav(wav, tmp.name)
            with open(tmp.name, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            os.remove(tmp.name)
        return {"audio_base64": audio_b64}

    elif job_type == "stt":
        audio_b64 = job_input.get("audio_base64", "")
        if not audio_b64:
            return {"error": "Missing 'audio_base64' input for STT"}
        print("[INFO] Transcribing audio from base64 input")
        return transcribe_from_base64(audio_b64)

    else:
        return {"error": "Invalid type. Use 'image', 'tts', or 'stt'."}

# ========== Start RunPod ==========
runpod.serverless.start({"handler": handler})
