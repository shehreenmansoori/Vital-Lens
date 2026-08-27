import os
import elevenlabs
import subprocess       #will help interact with CLI
import platform         #to check which platform its running on
from gtts import gTTS
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

def text_to_speech_old(input_text,output_filepath):
    language = "en"

    audioobj = gTTS(
        text=input_text,
        lang= language,
        slow=False
    )
    audioobj.save(output_filepath)

input_text = "HI this is user!"
#text_to_speech_old(input_text=input_text,output_filepath="gtts_testing.mp3")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def text_to_speech_labs_old(input_text,output_filepath):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text= input_text,
        voice = "EXAVITQu4vr4xnSDxMaL",
        output_format= "mp3_22050_32",
        model= "eleven_turbo_v2"
    )
    elevenlabs.save(audio,output_filepath)

#text_to_speech_labs_old(input_text,output_filepath="elevenlabs_testing.mp3")

#NEW
def text_to_speech_with_gtts(input_text, output_filepath):
    language = "en"
    audioobj = gTTS(text=input_text, lang=language, slow=False)
    audioobj.save(output_filepath)
    try:
        subprocess.run(
            ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', output_filepath],
            check=True
        )
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")

input_text="Hi this is Ai user, autoplay testing!"
#text_to_speech_with_gtts(input_text=input_text, output_filepath="gtts_testing_autoplay.mp3")

def text_to_speech_with_elevenlabs(input_text, output_filepath):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="EXAVITQu4vr4xnSDxMaL",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)
    try:
        subprocess.run(
            ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', output_filepath],
            check=True
        )
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")

#text_to_speech_with_elevenlabs(input_text=input_text, output_filepath="elevenlabs_testing_autoplay.mp3")
