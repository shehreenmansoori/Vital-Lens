import os
import base64 #converts bits into strings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#image_path = "acne.jpg"
def encode_image(image_path):
    image_file = open(image_path,"rb") #rb = read binary, reads the image
    return base64.b64encode(image_file.read()).decode('utf-8')


query = "Is there something wrong with my face?"
model = "qwen/qwen3.6-27b"

def analyze_image(query,model,encoded_image):
    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {
            "role": "system",
            "content": """
    You are MediAI, an AI medical image assistant. Analyze the provided image for visible signs of a possible health condition and respond in exactly one concise paragraph. Describe only what is visibly present, state the most likely possible condition without claiming certainty, briefly explain the visual features supporting that possibility, include a confidence score as a percentage, suggest safe home care only when appropriate, mention important red flags that require urgent medical attention, and clearly state that the response is not a confirmed medical diagnosis and that a qualified healthcare professional should be consulted. Do not use headings, bullet points, numbered lists, separate sections, line breaks, or step-by-step formatting. Keep the entire response brief, clear, cautious, and contained within a single paragraph, respond in less than 100 words.
    """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]
    chat_completion = client.chat.completions.create(
        messages= messages,
        model = model
    )
    return chat_completion.choices[0].message.content