import cv2 # OpenCV library for image processing
import pytesseract # Python wrapper for Tesseract OCR
import os # For handling file paths (like joining folder and file names)

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
    Extracts text from a given image file using Tesseract OCR.
    This is our first, basic OCR function!

    Args:
        image_path (str): The full path to the image file.

    Returns:
        str: The extracted text, or None if an error occurs.
    """
    print(f"--- Starting OCR for image: {image_path} ---")

    # 1. Check if the image file actually exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'. Please check the path and filename.")
        return None

    # 2. Load the image using OpenCV
    # cv2.imread() reads an image from the specified file.
    # It loads images by default in BGR (Blue, Green, Red) color order.
    image = cv2.imread(image_path)

    # 3. Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from '{image_path}'. The file might be corrupted or not a valid image.")
        return None

    # 4. Convert the image to RGB color order
    # Tesseract (the underlying OCR engine) typically expects images in RGB format,
    # but OpenCV loads them as BGR. This conversion is crucial for correct color
    # interpretation by Tesseract, which helps in accurate text recognition.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 5. Run Tesseract OCR on the image
    # pytesseract.image_to_string() is the core function here. It sends the image
    # to the Tesseract engine and gets back the recognized text.
    try:
        extracted_text = pytesseract.image_to_string(rgb_image)
        print("\n" + "="*60)
        print("Extracted Text:")
        print("="*60)
        print(extracted_text)
        print("="*60 + "\n")
        return extracted_text
    except pytesseract.TesseractNotFoundError:
        print("\n" + "="*60)
        print("ERROR: Tesseract OCR engine not found!")
        print("Please ensure Tesseract is installed on your system and its path is correctly configured.")
        print("Refer to Phase 1, Step 1.8 instructions.")
        print("If you installed it, you might need to specify its path in this script (see comments at the top).")
        print("="*60 + "\n")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during OCR: {e}")
        return None


if __name__ == "__main__":
    # --- Testing our function ---
    # Make sure you've put a sample image (e.g., receipt.jpg) in your 'samples' folder.
    # If your image has a different name or extension, change it here!
    sample_image_name = "receipt.jpg" # <--- IMPORTANT: Change this if your image has a different name!
    sample_image_path = os.path.join("samples", sample_image_name) # This correctly builds the path

    # Call our OCR function
    text_result = extract_text_from_image(sample_image_path)

    if text_result:
        # Optional: Save the extracted text to a file in the output folder
        output_filename = f"extracted_{os.path.splitext(sample_image_name)[0]}.txt"
        output_filepath = os.path.join("output", output_filename)
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(text_result)
            print(f"Successfully saved extracted text to: {output_filepath}")
        except Exception as e:
            print(f"Error saving extracted text to file: {e}")
    else:
        print("No text was extracted or an error occurred.") 