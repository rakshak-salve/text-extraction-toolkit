import cv2 # OpenCV library for image processing
import pytesseract # Python wrapper for Tesseract OCR
import os # For handling file paths (like joining folder and file names)
from preprocessing import preprocess_image  # <-- Import the preprocessing function
from text_detection import detect_text_regions

# --- IMPORTANT NOTE FOR WINDOWS USERS (and sometimes macOS/Linux if Tesseract isn't in PATH) ---
# If you installed Tesseract but get a "TesseractNotFoundError", it means Python
# can't find the Tesseract executable program. You need to tell pytesseract
# exactly where it is.
#
# Uncomment the line below and change the path to where your tesseract.exe is located.
# Common path for Windows: C:\Program Files\Tesseract-OCR\tesseract.exe
# Example:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
# If you're on macOS/Linux and installed via Homebrew/apt, it's usually in your PATH
# automatically, so you might not need this line. But if you get an error, check
# if it's in /usr/local/bin/tesseract or similar.
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
# -----------------------------------------------------------------------------------

def extract_text_from_image(image_path):
    """
    Extracts text from a given image file using Tesseract OCR,
    now with text detection and image preprocessing for improved accuracy.
    """
    print(f"--- Starting Advanced OCR for image: {image_path} ---")

    # 1. Check if the image file actually exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'. Please check the path and filename.")
        return None

    # 2. Load the original image for preprocessing
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load original image from '{image_path}'. Check file integrity.")
        return None

    # --- NEW STEP 1: Perform text detection ---
    print("Performing text detection...")
    regions, _, detected_img_for_display = detect_text_regions(original_image, min_confidence=0.7)

    if regions is None:
        print("Text detection failed, cannot proceed with OCR.")
        return None

    if not regions:
        print("No text regions detected in the image. No text to extract.")
        return ""

    output_detection_filename = f"detected_regions_for_ocr_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
    output_detection_filepath = os.path.join("output", output_detection_filename)
    cv2.imwrite(output_detection_filepath, detected_img_for_display)
    print(f"Saved detection visualization to: {output_detection_filepath}")

    all_extracted_text = []

    print(f"\nExtracting text from {len(regions)} detected regions...")
    for i, (startX, startY, endX, endY) in enumerate(regions):
        region_of_interest = original_image[startY:endY, startX:endX]
        if region_of_interest.size == 0:
            print(f"Skipping empty region {i+1}: ({startX}, {startY}, {endX}, {endY})")
            continue
        print(f" Processing region {i+1}/{len(regions)}: ({startX}, {startY}, {endX}, {endY})")
        processed_region = preprocess_image(region_of_interest)
        if processed_region is None:
            print(f"   Preprocessing failed for region {i+1}, skipping.")
            continue
        try:
            text = pytesseract.image_to_string(processed_region, config='--oem 3 --psm 6')
            clean_text = text.strip()
            if clean_text:
                all_extracted_text.append(clean_text)
            print(f"   Extracted: \"{clean_text[:50]}...\"" if len(clean_text) > 50 else f"   Extracted: \"{clean_text}\"")
        except pytesseract.TesseractNotFoundError:
            print("\n" + "="*60)
            print("ERROR: Tesseract OCR engine not found! Ensure it's installed and path is correct.")
            print("="*60 + "\n")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during OCR for region {i+1}: {e}")
            pass

    final_text_output = "\n\n".join(all_extracted_text)

    print("\n" + "="*60)
    print("FINAL EXTRACTED TEXT (from all detected regions):")
    print("="*60)
    print(final_text_output)
    print("="*60 + "\n")
    return final_text_output


if __name__ == "__main__":
    # --- Testing our function ---
    # Make sure you've put a sample image (e.g., receipt.jpg) in your 'samples' folder.
    # If your image has a different name or extension, change it here!
    sample_image_name = "book_page.jpg" # <--- IMPORTANT: Change this if your image has a different name!
    sample_image_path = os.path.join("samples", sample_image_name) # This correctly builds the path

    # Call our OCR function
    text_result = extract_text_from_image(sample_image_path)

    if text_result:
        # Optional: Save the extracted text to a file in the output folder
        output_filename = f"extracted_advanced_{os.path.splitext(os.path.basename(sample_image_path))[0]}.txt"
        output_filepath = os.path.join("output", output_filename)
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(text_result)
            print(f"Successfully saved final extracted text to: {output_filepath}")
        except Exception as e:
            print(f"Error saving final extracted text to file: {e}")
    else:
        print("No text was extracted, or an error occurred during advanced OCR.") 