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

# --- Sidebar Help & Info ---
st.sidebar.header("ℹ️ How to Use This App")
st.sidebar.markdown("""
- **Image Upload OCR:**
    - Upload a clear image (JPG/PNG) with printed text.
    - Click **Perform OCR** to see detected regions and extracted text.
- **Webcam OCR:**
    - For real-time OCR, run the script in your terminal (see instructions tab).
- **Tips:**
    - Use well-lit, sharp images for best results.
    - For errors, check the sidebar or see the terminal for details.
    - If you get 'Text detection failed', make sure the model file is in the 'models' folder.
""")
# Sidebar: Detection confidence slider
st.sidebar.markdown("---")
confidence = st.sidebar.slider(
    "Detection Confidence (higher = fewer, more certain boxes)",
    min_value=0.3, max_value=0.95, value=0.7, step=0.01,
    help="Lower for more boxes, higher for stricter detection. Default: 0.7"
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Rakshak")
st.sidebar.markdown(f"Current Time (IST): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

# --- Helper Function to Extract Text (with improved feedback) ---
def extract_text_from_image_array(image_array, display_detected_regions=False, min_confidence=0.7):
    """
    Extracts text from a NumPy image array using Tesseract OCR,
    with text detection and preprocessing.
    """
    st.info("OCR will find text regions, clean them up, and extract text. This may take a few seconds.")
    if len(image_array.shape) == 2:
        original_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    elif image_array.shape[2] == 4:
        original_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    else:
        original_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    st.subheader("Step 1: Detecting Text Regions")
    regions, _, detected_img_display = detect_text_regions(original_image, min_confidence=min_confidence)
    if regions is None:
        st.error("❌ Text detection failed. Please check that the model file exists in the 'models' folder. See sidebar for help.")
        return None, None, []
    if not regions:
        st.warning("⚠️ No text regions detected. Try a clearer image or check your lighting.")
        return "", detected_img_display, []
    st.success(f"✅ Detected {len(regions)} text region(s).")
    if display_detected_regions:
        st.image(detected_img_display, channels="BGR", caption="Detected Text Regions", use_container_width=True)
    all_extracted_text = []
    st.subheader("Step 2: Extracting Text from Regions (OCR)")
    progress_text = st.empty()
    progress_bar = st.progress(0)
    import pytesseract
    for i, (startX, startY, endX, endY) in enumerate(regions):
        progress = (i + 1) / len(regions)
        progress_bar.progress(progress)
        progress_text.text(f"Processing region {i+1} of {len(regions)}...")
        roi = original_image[startY:endY, startX:endX]
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            all_extracted_text.append("")
            continue
        processed_roi = preprocess_image(roi)
        if processed_roi is None:
            all_extracted_text.append("")
            continue
        try:
            text = pytesseract.image_to_string(processed_roi, config='--oem 3 --psm 7')
            clean_text = text.strip()
            all_extracted_text.append(clean_text)
        except Exception as e:
            all_extracted_text.append("")
    progress_text.text("OCR processing complete!")
    progress_bar.empty()
    final_text = "\n\n".join([t for t in all_extracted_text if t])
    return final_text, detected_img_display, all_extracted_text

# --- Real-time OCR Instructions ---
def show_realtime_instructions():
    st.header("Real-time Webcam OCR (Instructions)")
    st.markdown("""
    Streamlit does not directly support real-time webcam processing in the browser without advanced WebRTC integration.
    
    **To use real-time OCR:**
    1. Open your terminal.
    2. Activate your virtual environment.
    3. Run:
    """)
    st.code("python src/realtime_ocr.py")
    st.markdown("- Make sure your webcam is connected and not in use by another app.")
    st.info("The real-time script will show detected text live on your video feed. Press 'q' to quit, 's' to save a snapshot.")

# --- Main UI Layout ---
st.title("📄 Text Extraction Toolkit")
st.markdown("""
Welcome! This simple web app lets you extract text from images using OCR and computer vision.
- **Step 1:** Upload a clear image with printed text.
- **Step 2:** Click **Perform OCR** to see detected regions and extracted text.
- **Step 3:** (Optional) See instructions for real-time webcam OCR in the sidebar.
""")

app_mode = st.sidebar.radio(
    "Choose an OCR Mode:",
    ["Image Upload OCR", "Real-time Webcam OCR Instructions"]
)

if app_mode == "Image Upload OCR":
    st.header("🖼️ Image Upload OCR")
    st.markdown("Upload a **JPG** or **PNG** image with clear, printed text. For best results, use a well-lit, sharp image.")
    uploaded_file = st.file_uploader("Choose an image to analyze:", type=["jpg", "jpeg", "png"], help="Upload a clear image with printed text.")
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        image_np = np.array(Image.open(uploaded_file).convert('RGB'))
        st.success("Image uploaded successfully! Now click 'Perform OCR' below.")
        if st.button("Perform OCR", help="Detect text regions and extract text from the uploaded image."):
            st.write("---")
            with st.spinner("Processing image and extracting text..."):
                extracted_text, detected_img, all_regions_text = extract_text_from_image_array(
                    image_np, display_detected_regions=True, min_confidence=confidence)
            if extracted_text is not None:
                st.subheader("📋 Extracted Text:")
                if extracted_text:
                    for idx, region_text in enumerate(all_regions_text):
                        if region_text.strip():
                            st.markdown(f"**Region {idx+1}:**")
                            st.code(region_text)
                    st.download_button("Download All Text", extracted_text, file_name="extracted_text.txt")
                else:
                    st.info("No readable text found in the image.")
            st.write("---")
    else:
        st.info("⬆️ Upload an image to get started.")

elif app_mode == "Real-time Webcam OCR Instructions":
    show_realtime_instructions() 