"""AI Doctor with Vision and Voice — Gradio web application."""

import os

import gradio as gr
from dotenv import load_dotenv

from brain import analyze_image, encode_image
from doctor import text_to_speech
from voice import transcribe

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGES = {
    "Try sample: Acne": os.path.join(BASE_DIR, "acne.jpg"),
    "Try sample: Dandruff": os.path.join(BASE_DIR, "Dandruff.jpg"),
}


def process_inputs(
    audio_filepath, image_filepath, symptom_text, groq_key, eleven_key,
    progress=gr.Progress(),
):
    """Pipeline: transcribe (or read typed text) -> analyze image -> speak.

    Keys typed into the API Settings panel take priority; blank fields
    fall back to the server environment.
    """
    groq_key = (groq_key or "").strip()
    eleven_key = (eleven_key or "").strip()

    # Any audio or image needs Groq (STT and/or vision) — check before
    # calling out so a missing key is a friendly message, not an error.
    if (audio_filepath or image_filepath) and not (
        groq_key or os.environ.get("GROQ_API_KEY")
    ):
        return (
            "",
            "I'm missing a Groq API key. Open **API Settings** below, paste "
            "your own key (free at console.groq.com/keys), and try again — "
            "or ask the host to set one on the server.",
            None,
        )

    progress(0.1, desc="Listening...")

    speech_to_text_output = transcribe(audio_filepath, groq_api_key=groq_key)

    # Spoken question wins; typed text is the fallback when there's no audio.
    if speech_to_text_output:
        user_query = speech_to_text_output
    else:
        user_query = (symptom_text or "").strip()

    progress(0.4, desc="Analyzing...")

    if image_filepath:
        doctor_response = analyze_image(
            query=user_query if user_query else "What do you see and is anything wrong medically?",
            encoded_image=encode_image(image_filepath),
            groq_api_key=groq_key,
        )
    elif user_query:
        doctor_response = "No image provided for me to analyze"
    else:
        # Nothing to work with — no API calls, no crash.
        return (
            speech_to_text_output,
            "Please record your voice, type your symptoms, or upload an image "
            "so I can take a look.",
            None,
        )

    progress(0.7, desc="Generating the doctor's voice...")

    voice_of_doctor = text_to_speech(
        doctor_response, elevenlabs_api_key=eleven_key
    )

    progress(1.0, desc="Done")
    return speech_to_text_output, doctor_response, voice_of_doctor


with gr.Blocks(title="AI Doctor with Vision and Voice") as demo:
    gr.Markdown("# AI Doctor with Vision and Voice")
    gr.Markdown(
        "Describe your symptoms by voice or text and upload an image — "
        "the AI doctor will respond in writing and speech. "
        "*For educational purposes only, not a medical diagnosis.*"
    )

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Your Voice")
        image_input = gr.Image(type="filepath", label="Medical Image")

    text_input = gr.Textbox(
        label="Or type your symptoms / question",
        placeholder="e.g. Is there something wrong with my face?",
    )

    with gr.Row():
        for sample_name, sample_path in SAMPLE_IMAGES.items():
            gr.Button(sample_name).click(
                fn=lambda p=sample_path: p, outputs=image_input
            )

    submit = gr.Button("Consult the Doctor", variant="primary")

    speech_output = gr.Textbox(label="Speech to Text")
    response_output = gr.Textbox(label="Doctor's Response")
    audio_output = gr.Audio(label="Doctor's Voice")

    with gr.Accordion("API Settings (optional — bring your own key)", open=False):
        groq_key_input = gr.Textbox(
            type="password",
            label="Groq API Key",
            placeholder="gsk_... (leave blank to use the server key)",
        )
        eleven_key_input = gr.Textbox(
            type="password",
            label="ElevenLabs API Key (optional)",
            placeholder="Leave blank to use the server key or the free gTTS voice",
        )

    submit.click(
        fn=process_inputs,
        inputs=[audio_input, image_input, text_input, groq_key_input, eleven_key_input],
        outputs=[speech_output, response_output, audio_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
        debug=False,
    )
