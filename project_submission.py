





import streamlit as st

st.title("Language Translation with Text-to-Speech")
st.write("This application allows you to translate text from uploaded files or direct input and convert the translated text to speech.")

input_method = st.radio("Choose input method", ("Upload a file", "Enter text directly"))

uploaded_file = None
input_text = ""

if input_method == "Upload a file":
    uploaded_file = st.file_uploader("Upload a file", type=['pdf', 'txt', 'xlsx', 'xls', 'csv'])
elif input_method == "Enter text directly":
    input_text = st.text_area("Enter text here:")

import pandas as pd
import PyPDF2
import io

def extract_text_from_file(uploaded_file):
    """Extracts text content from uploaded files."""
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        try:
            if file_extension == 'pdf':
                reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
                return text
            elif file_extension == 'txt':
                return uploaded_file.getvalue().decode("utf-8")
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
                return df.to_string()
            elif file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
                return df.to_string()
            else:
                return "Unsupported file type."
        except Exception as e:
            return f"Error processing file: {e}"
    return ""

target_languages = ["English", "Spanish", "French", "German", "Chinese"]
target_language = st.selectbox("Select Target Language", target_languages)

import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

translation_template = """Translate the following text into {target_language}:

{text}
"""
translation_prompt = PromptTemplate(
    input_variables=["text", "target_language"],
    template=translation_template,
)

llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

def translate_text(text: str, target_language: str) -> str:
    """Translates the input text to the target language."""
    translated_text = translation_chain.invoke({"text": text, "target_language": target_language})
    return translated_text['text']

from gtts import gTTS
import io

def text_to_speech(text: str, lang: str) -> bytes:
    """Converts text to speech using gTTS."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        return audio_bytes.getvalue()
    except Exception as e:
        st.error(f"Error during text-to-speech conversion: {e}")
        return None

# Check if either a file is uploaded or text is entered
processed_text = ""
if input_method == "Upload a file" and uploaded_file is not None:
    processed_text = extract_text_from_file(uploaded_file)
elif input_method == "Enter text directly" and input_text:
    processed_text = input_text

translated_text = ""
audio_bytes = None

if processed_text:
    st.subheader("Original Text")
    st.write(processed_text)

    st.subheader("Translated Text")
    # Assuming target_language is already defined from the selectbox
    translated_text = translate_text(processed_text, target_language)
    st.write(translated_text)

    if translated_text:
        # Convert the target language name to a language code for gTTS
        # This is a simplified mapping, you might need a more comprehensive one
        lang_code_map = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Chinese": "zh-CN" # Example, adjust as needed
        }
        target_lang_code = lang_code_map.get(target_language, "en") # Default to English

        audio_bytes = text_to_speech(translated_text, target_lang_code)

        if audio_bytes:
            st.subheader("Audio Playback")
            st.audio(audio_bytes, format="audio/mp3")

            st.subheader("Download Audio")
            st.download_button(
                label="Download Audio",
                data=audio_bytes,
                file_name="translated_audio.mp3",
                mime="audio/mp3"
            )

import nbformat

# Path of the current notebook (Colab provides it only when saving locally)
notebook_path = "/content/.ipynb"
output_path = "/content/your_notebook_clean.py"

# Load the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Extract only code cells
code_cells = [cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"]

# Save to .py
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(code_cells))

print(f"Saved to {output_path}")
