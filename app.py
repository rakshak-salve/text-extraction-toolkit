import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from preprocessing import preprocess_image
from text_detection import detect_text_regions

# --- IMPORTANT NOTE FOR WINDOWS USERS (Tesseract Path) ---
# If you needed to set pytesseract.pytesseract.tesseract_cmd in basic_ocr.py/realtime_ocr.py,
# you MUST set it here as well for the Streamlit app to find Tesseract.
# Example:
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# --------------------------------------------------------

st.set_page_config(
    page_title="Text Extraction Toolkit",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def extract_text_from_image_array(image_array, display_detected_regions=False):
    """
    Extracts text from a NumPy image array using Tesseract OCR,
    with text detection and preprocessing.
    """
    st.info("Starting OCR process...")
    if len(image_array.shape) == 2:
        original_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    elif image_array.shape[2] == 4:
        original_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    else:
        original_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    st.subheader("1. Text Detection")
    regions, _, detected_img_display = detect_text_regions(original_image, min_confidence=0.7)
    if regions is None:
        st.error("Text detection failed. Please check the model path and file integrity.")
        return None, None
    if not regions:
        st.warning("No text regions detected in the image.")
        return "", detected_img_display
    st.success(f"Detected {len(regions)} text regions.")
    if display_detected_regions:
        st.image(detected_img_display, channels="BGR", caption="Detected Text Regions", use_column_width=True)
    all_extracted_text = []
    st.subheader("2. Text Extraction (OCR)")
    progress_text = st.empty()
    progress_bar = st.progress(0)
    import pytesseract
    for i, (startX, startY, endX, endY) in enumerate(regions):
        progress = (i + 1) / len(regions)
        progress_bar.progress(progress)
        progress_text.text(f"Processing region {i+1}/{len(regions)}...")
        roi = original_image[startY:endY, startX:endX]
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            continue
        processed_roi = preprocess_image(roi)
        if processed_roi is None:
            continue
        try:
            text = pytesseract.image_to_string(processed_roi, config='--oem 3 --psm 7')
            clean_text = text.strip()
            if clean_text:
                all_extracted_text.append(clean_text)
        except Exception as e:
            pass
    progress_text.text("OCR processing complete!")
    progress_bar.empty()
    final_text = "\n\n".join(all_extracted_text)
    return final_text, detected_img_display

def show_realtime_instructions():
    st.warning("Streamlit does not directly support real-time webcam processing in the browser without advanced WebRTC integration (beyond the scope of this simple frontend).")
    st.info("To use the real-time webcam OCR, please run the following command in your terminal (after activating your virtual environment):")
    st.code("python src/realtime_ocr.py")
    st.markdown("Ensure your webcam is connected and not in use by another application.")

st.title("📄 Text Extraction Toolkit")
st.markdown("Extract text from images or use real-time OCR!")

st.sidebar.header("Navigation")
app_mode = st.sidebar.radio(
    "Choose an OCR Mode:",
    ["Image Upload OCR", "Real-time Webcam OCR Instructions"]
)

if app_mode == "Image Upload OCR":
    st.header("Image Upload OCR")
    st.markdown("Upload an image file (PNG, JPG) to extract text.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        image_bytes = uploaded_file.read()
        image_np = np.array(Image.open(uploaded_file).convert('RGB'))
        st.write("Image uploaded successfully!")
        if st.button("Perform OCR"):
            st.write("---")
            with st.spinner("Processing image and extracting text..."):
                extracted_text, detected_img = extract_text_from_image_array(image_np, display_detected_regions=True)
            if extracted_text is not None:
                st.subheader("Extracted Text:")
                if extracted_text:
                    st.code(extracted_text)
                else:
                    st.info("No readable text found in the image.")
            st.write("---")
elif app_mode == "Real-time Webcam OCR Instructions":
    st.header("Real-time Webcam OCR (Instructions)")
    st.markdown("This section provides instructions on how to run the real-time webcam OCR using the separate `realtime_ocr.py` script.")
    show_realtime_instructions()

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Rakshak")
st.sidebar.markdown(f"Current Time (IST): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}") 