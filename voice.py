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


def _get_client(api_key=None):
    """Resolve the Groq client.

    A user-supplied key (bring-your-own-key) returns a fresh, uncached
    client so concurrent requests with different keys never share state;
    without one, a single client built from the server environment is
    created once and reused across requests.
    """
    api_key = (api_key or "").strip()
    if api_key:
        return Groq(api_key=api_key)

    global _client
    with _client_lock:
        if _client is None:
            env_key = os.environ.get("GROQ_API_KEY")
            if not env_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to .env, the hosting "
                    "environment, or the API Settings panel in the app."
                )
            _client = Groq(api_key=env_key)
        return _client


def transcribe(audio_filepath, stt_model=STT_MODEL, groq_api_key=None):
    """Transcribe an audio file to text.

    Returns "" when no audio was provided (or transcription fails), so
    image-only and text-only flows complete without crashing the request.
    """
    if not audio_filepath:
        logging.info("No audio provided, skipping transcription.")
        return ""

    try:
        with open(audio_filepath, "rb") as audio_file:
            transcription = _get_client(groq_api_key).audio.transcriptions.create(
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
