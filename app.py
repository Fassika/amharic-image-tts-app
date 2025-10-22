import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import tempfile
import os
from gtts import gTTS

def generate_audio_bytes(text, speed):
    """Generate audio using gTTS (Amharic support)."""
    # gTTS doesn't support fine-grained speed; approximate with slow=True for slower speeds
    slow = speed < 1.0
    tts = gTTS(text=text, lang='am', slow=slow)
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tts.save(tmp_path)
        with open(tmp_path, 'rb') as f:
            audio_bytes = f.read()
        os.remove(tmp_path)
    return audio_bytes

# Streamlit app
st.title("Amharic Text-to-Speech from Image")
st.info("Internet connection required for audio generation using Google TTS.")

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
    # Display the original image
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
            
            speed = st.slider("Reading Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            st.info("Note: Speed control is approximate (slower than 1.0 uses a slower TTS rate; faster uses normal rate).")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Read"):
                    st.write("Extracted text:", extracted_text)
                    st.write("Speed:", speed)
                    with st.spinner("Generating audio..."):
                        try:
                            audio_bytes = generate_audio_bytes(extracted_text, speed)
                            st.audio(audio_bytes, format="audio/mp3")
                        except Exception as e:
                            st.error(f"Failed to generate audio: {str(e)}")
            
            with col2:
                if st.button("Save to File"):
                    with st.spinner("Generating audio..."):
                        try:
                            audio_bytes = generate_audio_bytes(extracted_text, speed)
                            default_filename = f"amharic_audio_{speed}x.mp3"
                            st.download_button(
                                label="Download Audio",
                                data=audio_bytes,
                                file_name=default_filename,
                                mime="audio/mp3"
                            )
                        except Exception as e:
                            st.error(f"Failed to generate audio: {str(e)}")
else:
    st.info("Please provide an image to proceed.")
