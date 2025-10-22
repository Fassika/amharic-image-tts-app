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

def process_image(img, brightness=0, contrast=1.0, crop_left=0.0, crop_top=0.0, crop_right=0.0, crop_bottom=0.0):
    """Apply cropping, then brightness and contrast adjustments to the image."""
    height, width = img.shape[:2]
    
    # Apply cropping
    x1 = int(width * crop_left)
    y1 = int(height * crop_top)
    x2 = int(width * (1 - crop_right))
    y2 = int(height * (1 - crop_bottom))
    
    # Ensure valid crop bounds
    x1 = max(0, min(x1, x2, width))
    y1 = max(0, min(y1, y2, height))
    x2 = max(x1, min(x2, width))
    y2 = max(y1, min(y2, height))
    
    cropped = img[y1:y2, x1:x2]
    
    if cropped.size == 0:
        return img  # Fallback to original if crop is invalid
    
    # Apply brightness and contrast
    adjusted = cv2.convertScaleAbs(cropped, alpha=contrast, beta=brightness)
    return adjusted

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
    
    # Initialize session state for editing
    if 'show_editor' not in st.session_state:
        st.session_state.show_editor = False
    if 'edited_img' not in st.session_state:
        st.session_state.edited_img = None
    if 'brightness' not in st.session_state:
        st.session_state.brightness = 0
    if 'contrast' not in st.session_state:
        st.session_state.contrast = 1.0
    if 'crop_left' not in st.session_state:
        st.session_state.crop_left = 0.0
    if 'crop_top' not in st.session_state:
        st.session_state.crop_top = 0.0
    if 'crop_right' not in st.session_state:
        st.session_state.crop_right = 0.0
    if 'crop_bottom' not in st.session_state:
        st.session_state.crop_bottom = 0.0
    
    # Button to toggle editor
    if st.button("Edit Image"):
        st.session_state.show_editor = not st.session_state.show_editor
    
    if st.session_state.show_editor:
        with st.expander("Image Editor", expanded=True):
            # Brightness and Contrast
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.brightness = st.slider("Brightness", -100, 100, st.session_state.brightness)
            with col2:
                st.session_state.contrast = st.slider("Contrast", 0.1, 3.0, st.session_state.contrast, step=0.1)
            
            # Cropping
            crop_col1, crop_col2 = st.columns(2)
            with crop_col1:
                st.session_state.crop_left = st.slider("Crop Left (%)", 0.0, 50.0, st.session_state.crop_left, step=1.0) / 100.0
                st.session_state.crop_top = st.slider("Crop Top (%)", 0.0, 50.0, st.session_state.crop_top, step=1.0) / 100.0
            with crop_col2:
                st.session_state.crop_right = st.slider("Crop Right (%)", 0.0, 50.0, st.session_state.crop_right, step=1.0) / 100.0
                st.session_state.crop_bottom = st.slider("Crop Bottom (%)", 0.0, 50.0, st.session_state.crop_bottom, step=1.0) / 100.0
            
            if st.button("Apply Edits"):
                image_bytes = image.getvalue()
                np_arr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                st.session_state.edited_img = process_image(
                    img,
                    brightness=st.session_state.brightness,
                    contrast=st.session_state.contrast,
                    crop_left=st.session_state.crop_left,
                    crop_top=st.session_state.crop_top,
                    crop_right=st.session_state.crop_right,
                    crop_bottom=st.session_state.crop_bottom
                )
                st.rerun()
            
            # Show preview if edits applied
            if st.session_state.edited_img is not None:
                st.image(st.session_state.edited_img, caption="Edited Image Preview", use_container_width=True)
    
    # Use edited image if available, else original
    if st.session_state.edited_img is not None:
        img_to_process = st.session_state.edited_img
    else:
        image_bytes = image.getvalue()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_to_process = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img_to_process is None:
        st.error("Failed to load image. Please try another file.")
    else:
        gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
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
