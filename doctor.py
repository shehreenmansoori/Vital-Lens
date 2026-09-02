"""Doctor's voice: text-to-speech with ElevenLabs and automatic gTTS fallback."""

import logging
import os
import tempfile
import threading

from dotenv import load_dotenv
from elevenlabs import save
from elevenlabs.client import ElevenLabs
from gtts import gTTS

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
TTS_MODEL = "eleven_turbo_v2"
OUTPUT_FORMAT = "mp3_22050_32"
REQUEST_TIMEOUT_SECONDS = 60

_client = None
_client_lock = threading.Lock()


def _get_client():
    """Return the shared ElevenLabs client, built once from the server
    environment. Returns None when no key is available, so gTTS takes over."""
    global _client
    with _client_lock:
        if _client is None:
            env_key = os.environ.get("ELEVENLABS_API_KEY")
            if not env_key:
                return None
            _client = ElevenLabs(api_key=env_key)
        return _client


def _new_audio_path():
    """Unique temp file path per request — safe for concurrent users."""
    handle = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    handle.close()
    return handle.name


def text_to_speech(input_text):
    """Synthesize speech from text and return the audio filepath.

    Primary: ElevenLabs. Falls back to gTTS on any failure (quota, rate
    limit, missing key). Returns None only if both engines fail, so the
    text response can still be shown without crashing the request.
    """
    if not input_text:
        logging.warning("No text provided for speech synthesis.")
        return None

    output_filepath = _new_audio_path()

    client = _get_client()
    if client is not None:
        try:
            audio = client.text_to_speech.convert(
                text=input_text,
                voice_id=VOICE_ID,
                model_id=TTS_MODEL,
                output_format=OUTPUT_FORMAT,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            save(audio, output_filepath)
            logging.info("ElevenLabs voice generated.")
            return output_filepath
        except Exception as e:
            logging.warning(
                "ElevenLabs failed (%s), falling back to gTTS.", e
            )

    try:
        gTTS(text=input_text, lang="en", slow=False).save(output_filepath)
        logging.info("gTTS voice generated.")
        return output_filepath
    except Exception as e:
        logging.error("Text-to-speech failed entirely: %s", e)
        return None
