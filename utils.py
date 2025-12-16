import streamlit as st
from PIL import Image
import pytesseract


@st.cache_resource
def load_whisper_model():

    import whisper 
    return whisper.load_model("base")

def process_image(image_file):
    try:
        image = Image.open(image_file)
        gray_image = image.convert('L')
        text = pytesseract.image_to_string(gray_image)
        return text.strip(), None
    except Exception as e:
        return "", str(e)

def process_audio(audio_file_path):
    try:

        model = load_whisper_model()
        result = model.transcribe(audio_file_path)
        return result["text"].strip(), None
    except Exception as e:
        return "", str(e)