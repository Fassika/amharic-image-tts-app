import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from gtts import gTTS
import io
from pdf2image import convert_from_bytes



def preprocess_image(img):
    """Preprocess image for better OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    return thresh

def extract_text_from_image(pil_image):
    """Extract Amharic text from PIL image using Tesseract."""
    return pytesseract.image_to_string(pil_image, lang='amh').strip()

def generate_audio_bytes(text, speed):
    """Generate audio using gTTS and return as bytes. Speed is approximated with slow mode."""
    slow = speed < 1.0
    tts = gTTS(text, lang='am', slow=slow)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

# Streamlit app
st.title("Amharic Text-to-Speech from Image or PDF")

input_method = st.selectbox("Choose input method", ["Upload Image File", "Upload PDF File", "Take Photo with Camera"])

extracted_text = ""
display_image = None

if input_method == "Upload Image File":
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Display the original image
        st.image(uploaded_file, caption="Input Image", use_container_width=True)
        
        image_bytes = uploaded_file.getvalue()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Failed to load image. Please try another file.")
        else:
            preprocessed = preprocess_image(img)
            pil_image = Image.fromarray(preprocessed)
            extracted_text = extract_text_from_image(pil_image)
            display_image = uploaded_file

elif input_method == "Upload PDF File":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        pdf_bytes = uploaded_file.getvalue()
        images = convert_from_bytes(pdf_bytes)
        display_image = images[0] if images else None
        
        extracted_text_parts = []
        for idx, img in enumerate(images):
            open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            preprocessed = preprocess_image(open_cv_image)
            pil_preprocessed = Image.fromarray(preprocessed)
            text = extract_text_from_image(pil_preprocessed)
            if text:
                extracted_text_parts.append(text)
        
        extracted_text = "\n\n".join(extracted_text_parts)

elif input_method == "Take Photo with Camera":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        # Display the original image
        st.image(camera_image, caption="Input Image", use_container_width=True)
        
        image_bytes = camera_image.getvalue()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Failed to load image. Please try another file.")
        else:
            preprocessed = preprocess_image(img)
            pil_image = Image.fromarray(preprocessed)
            extracted_text = extract_text_from_image(pil_image)
            display_image = camera_image

if display_image and input_method == "Upload PDF File":
    st.image(display_image, caption="First Page of PDF", use_container_width=True)

if extracted_text:
    if not extracted_text.strip():
        st.warning("No text extracted from the document. Try a clearer image or PDF.")
    else:
        st.subheader("Extracted Amharic Text")
        st.text_area("Text", extracted_text, height=150)
        
        speed = st.slider("Reading Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Read"):
                with st.spinner("Generating audio..."):
                    audio_bytes = generate_audio_bytes(extracted_text, speed)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.error("Failed to generate audio.")
        
        with col2:
            if st.button("Save to File"):
                with st.spinner("Generating audio..."):
                    audio_bytes = generate_audio_bytes(extracted_text, speed)
                if audio_bytes:
                    default_filename = f"amharic_audio_{speed}x.mp3"
                    st.download_button(
                        label="Download Audio",
                        data=audio_bytes,
                        file_name=default_filename,
                        mime="audio/mp3"
                    )
                else:
                    st.error("Failed to generate audio.")
else:
    st.info("Please provide an image or PDF to proceed.")
