"""Speech-to-text transcription via Groq Whisper."""

import logging
import os
import threading

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

STT_MODEL = "whisper-large-v3"
REQUEST_TIMEOUT_SECONDS = 60

_client = None
_client_lock = threading.Lock()


def _get_client():
    """Lazily create a single Groq client, reused across requests."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to .env or the hosting environment."
                )
            _client = Groq(api_key=api_key)
        return _client


def transcribe(audio_filepath, stt_model=STT_MODEL):
    """Transcribe an audio file to text.

    Returns "" when no audio was provided (or transcription fails), so
    image-only and text-only flows complete without crashing the request.
    """
    if not audio_filepath:
        logging.info("No audio provided, skipping transcription.")
        return ""

    try:
        with open(audio_filepath, "rb") as audio_file:
            transcription = _get_client().audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en",
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        logging.info("Transcription complete.")
        return transcription.text
    except RuntimeError:
        # Missing API key — surface this one, it is a configuration error.
        raise
    except Exception as e:
        logging.error("Speech-to-text failed, continuing without it: %s", e)
        return ""
