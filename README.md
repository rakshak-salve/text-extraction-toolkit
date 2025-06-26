# Text Extraction Toolkit

Extract text from images using OCR and deep learning. Use the web app or batch process your images easily.

## Quick Start
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Download the model:**
   - [Download EAST model](https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb)
   - Place it in `models/` as `EAST_text_detection.pb`
3. **Run the web app:**
   ```sh
   streamlit run app.py
   ```
   Or batch process images in `input/`:
   ```sh
   python process_user_images.py
   ```

## Troubleshooting
- **Model not found?** Download and place in `models/` as above.
- **No text found?** Try a clearer image or adjust confidence.
- **Tesseract not found?** Install it and add to PATH (see code comments).

---
Developed by Rakshak Salve.
