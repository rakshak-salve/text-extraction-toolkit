import cv2
import numpy as np
import os

# Define the path to the EAST text detector model
EAST_MODEL_PATH = os.path.join("..", "models", "EAST_text_detection.pb")

def detect_text_regions(image_path_or_array, min_confidence=0.5):
    """
    Detects text regions in an image using the EAST text detector.
    Returns (text_regions, original_image_copy, detection_image).
    """
    print(f"--- Starting text detection using EAST model: {EAST_MODEL_PATH} ---")
    if isinstance(image_path_or_array, str):
        if not os.path.exists(image_path_or_array):
            print(f"Error: Image file not found at '{image_path_or_array}'.")
            return None, None, None
        image = cv2.imread(image_path_or_array)
    else:
        image = image_path_or_array
    if image is None:
        print(f"Error: Could not load image for text detection.")
        return None, None, None
    original_image_copy = image.copy()
    (H, W) = image.shape[:2]
    try:
        net = cv2.dnn.readNet(EAST_MODEL_PATH)
    except Exception as e:
        print(f"Error: Could not load EAST text detection model from '{EAST_MODEL_PATH}'.")
        print(f"Please ensure the file exists and the path is correct. Error: {e}")
        return None, None, None
    output_layers = ["feature_fusion/Conv_7/Sigmoid", "geometry_output/feature_fusion/Conv_7/concat_dml4"]
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(output_layers)
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]
        for x in range(0, numCols):
            if scores_data[x] < min_confidence:
                continue
            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]
            endX = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            endY = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            confidences.append(scores_data[x])
            rects.append((startX, startY, endX, endY))
    indices = cv2.dnn.NMSBoxes(rects, confidences, min_confidence, 0.3)
    text_regions = []
    detection_image = original_image_copy.copy()
    if len(indices) > 0:
        for i in indices.flatten():
            (startX, startY, endX, endY) = rects[i]
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(W, endX)
            endY = min(H, endY)
            padding = 5
            startX = max(0, startX - padding)
            startY = max(0, startY - padding)
            endX = min(W, endX + padding)
            endY = min(H, endY + padding)
            text_regions.append((startX, startY, endX, endY))
            cv2.rectangle(detection_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    print(f"--- Detected {len(text_regions)} text regions. ---")
    return text_regions, original_image_copy, detection_image

if __name__ == "__main__":
    sample_image_name = "book_page.jpg"
    sample_image_path = os.path.join("samples", sample_image_name)
    print(f"Testing text detection on: {sample_image_path}")
    regions, original_img, detected_img = detect_text_regions(sample_image_path, min_confidence=0.7)
    if regions is not None and detected_img is not None:
        output_detection_name = f"detected_regions_{os.path.splitext(sample_image_name)[0]}.jpg"
        output_detection_path = os.path.join("output", output_detection_name)
        cv2.imwrite(output_detection_path, detected_img)
        print(f"Image with detected regions saved to: {output_detection_path}")
        cv2.imshow("Detected Text Regions", detected_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("\nDetected Regions (startX, startY, endX, endY):")
        for r in regions:
            print(r)
    else:
        print("Text detection failed. Check for errors above.") 