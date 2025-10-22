import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import edge_tts
import asyncio
import tempfile
import os


async def generate_audio_async(text, voice, rate_str, output_path):
    """Async function to generate audio using edge-tts."""
    communicate = edge_tts.Communicate(text, voice, rate=rate_str)
    await communicate.save(output_path)

def generate_audio_bytes(text, voice, speed):
    """Generate audio and return as bytes."""
    if speed == 1.0:
        rate_str = "+0%"
    elif speed > 1.0:
        rate_str = f"+{int((speed - 1) * 100)}%"
    else:
        rate_str = f"-{int((1 - speed) * 100)}%"

    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        asyncio.run(generate_audio_async(text, voice, rate_str, tmp_path))
        with open(tmp_path, 'rb') as f:
            audio_bytes = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    return audio_bytes

# Streamlit app
st.title("Amharic Text-to-Speech from Image")

input_method = st.selectbox("Choose input method", ["Upload Image File", "Take Photo with Camera"])

image = None
if input_method == "Upload Image File":
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = uploaded_file
elif input_method == "Take Photo with Camera":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = camera_image

if image:
    # Display the original image (updated line)
    st.image(image, caption="Input Image", use_container_width=True)
    
    image_bytes = image.getvalue()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Failed to load image. Please try another file.")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        
        pil_image = Image.fromarray(thresh)
        
        extracted_text = pytesseract.image_to_string(pil_image, lang='amh').strip()
        
        if not extracted_text:
            st.warning("No text extracted from the image. Try a clearer image.")
        else:
            st.subheader("Extracted Amharic Text")
            st.text_area("Text", extracted_text, height=150)
            
            voice_option = st.selectbox("Voice", ["Male", "Female"])
            voice_code = "am-ET-AmehaNeural" if voice_option == "Male" else "am-ET-MekdesNeural"
            
            speed = st.slider("Reading Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Read"):
                    st.write("Extracted text:", extracted_text)
                    st.write("Voice code:", voice_code)
                    st.write("Speed:", speed, "Rate:", rate_str)
                    with st.spinner("Generating audio..."):
                        audio_bytes = generate_audio_bytes(extracted_text, voice_code, speed)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        st.error("Failed to generate audio.")
            
            with col2:
                if st.button("Save to File"):
                    with st.spinner("Generating audio..."):
                        audio_bytes = generate_audio_bytes(extracted_text, voice_code, speed)
                    if audio_bytes:
                        default_filename = f"amharic_audio_{voice_option.lower()}_{speed}x.mp3"
                        st.download_button(
                            label="Download Audio",
                            data=audio_bytes,
                            file_name=default_filename,
                            mime="audio/mp3"
                        )
                    else:
                        st.error("Failed to generate audio.")
else:
    st.info("Please provide an image to proceed.")
