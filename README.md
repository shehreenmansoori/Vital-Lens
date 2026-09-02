# Medic — AI Doctor with Vision and Voice

A real-time, multimodal AI telehealth assistant: speak your symptoms, upload a medical image, and receive a spoken clinical assessment.

**Pipeline:** mic / typed text → **Groq Whisper** (`whisper-large-v3`) transcription → **Groq Vision** (`qwen/qwen3.6-27b`) image + symptom analysis → **ElevenLabs** (`eleven_turbo_v2`) spoken response, with automatic **gTTS** fallback.

---

## Features

- **Speech-to-text** via Groq Whisper, with a typed-symptom fallback
- **Vision-based analysis**: a single clinical system prompt produces one concise paragraph — likely condition, supporting features, confidence score, home care, and red flags
- **Voice synthesis** via ElevenLabs, falling back to gTTS if the key is missing or quota is exhausted
- **Gradio web UI**: microphone capture, image upload, and one-click sample images (acne, dandruff)
- **Production hardening**: thread-safe lazy API clients, 60s request timeouts, image downscaling before upload — runs comfortably on a 512MB free tier

## Project Structure

```text
app.py            # Gradio UI + pipeline orchestration
voice.py          # Speech-to-text (Groq Whisper)
brain.py          # Vision analysis (Groq qwen/qwen3.6-27b)
doctor.py         # Text-to-speech (ElevenLabs + gTTS fallback)
requirements.txt  # Pinned dependencies
.env.example      # API key template
acne.jpg          # Sample image
Dandruff.jpg      # Sample image
```

## Quick Start

Requires **Python 3.10+**. No system packages needed — all dependencies are pure Python or prebuilt wheels.

```bash
git clone https://github.com/shehreenmansoori/Medic.git
cd Medic
python -m venv .venv
.venv\Scripts\activate          # Windows (use: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env            # then fill in your keys
python app.py
```

Get your keys:

- [Groq API key](https://console.groq.com/keys) — required (STT + vision)
- [ElevenLabs API key](https://elevenlabs.io/) — optional; gTTS fallback covers TTS without it

Open `http://127.0.0.1:7860`, record your voice or type your symptoms, upload an image (or load a sample), and submit. You'll see the transcription, the doctor's response, and an audio player with the spoken diagnosis.

## Deploying to Render (Free Tier)

1. Create a new **Web Service** on [Render](https://render.com/) from this repository.
2. Configure:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `python app.py`
   - **Environment variables**: `GROQ_API_KEY`, `ELEVENLABS_API_KEY`
3. Deploy — the app binds to `0.0.0.0` on Render's dynamic `$PORT` automatically.

> Free-tier services spin down after 15 minutes of inactivity; the first request after idle has a short cold start.

## Medical Disclaimer

> [!WARNING]
> Medic is developed solely for educational, informational, and research purposes. It is **not** a certified medical diagnostic device and does **not** provide medical advice, diagnosis, or treatment. Always consult a qualified physician or licensed healthcare provider regarding any medical condition. Never disregard or delay professional medical advice because of output from this software.
