"""Vision-based medical image analysis via Groq."""

import base64
import io
import logging
import os
import threading

from dotenv import load_dotenv
from groq import Groq
from PIL import Image

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

VISION_MODEL = "qwen/qwen3.6-27b"
REQUEST_TIMEOUT_SECONDS = 60
MAX_IMAGE_DIMENSION = 1024

SYSTEM_PROMPT = """You are MediAI, an AI medical image assistant. Analyze the provided image for visible signs of a possible health condition and respond in exactly one concise paragraph. Describe only what is visibly present, state the most likely possible condition without claiming certainty, briefly explain the visual features supporting that possibility, include a confidence score as a percentage, suggest safe home care only when appropriate, mention important red flags that require urgent medical attention, and clearly state that the response is not a confirmed medical diagnosis and that a qualified healthcare professional should be consulted. If the user describes symptoms or asks a question, factor that into your assessment. Never start with phrases like 'In the image I see' — say 'With what I see, I think you have ...' instead. Do not use headings, bullet points, numbered lists, separate sections, line breaks, or step-by-step formatting. Respond as if talking to a real person, not as an AI model. Keep the entire response brief, clear, cautious, and contained within a single paragraph, in less than 100 words."""

_client = None
_client_lock = threading.Lock()


def _get_client():
    """Return the shared Groq client, built once from the server environment."""
    global _client
    with _client_lock:
        if _client is None:
            env_key = os.environ.get("GROQ_API_KEY")
            if not env_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to .env or the hosting "
                    "environment."
                )
            _client = Groq(api_key=env_key)
        return _client


def encode_image(image_path):
    """Read an image, downscale it, and return its base64-encoded string.

    Downscaling keeps the payload small (fast upload, low RAM) and safely
    under Groq's request size limit, without affecting diagnostic quality.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def analyze_image(query, encoded_image, model=VISION_MODEL):
    """Send the user's symptoms/question plus the image to the vision model."""
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    },
                },
            ],
        },
    ]

    chat_completion = _get_client().chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.3,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return chat_completion.choices[0].message.content
