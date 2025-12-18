import cv2
from src.detect_image import detect_emotion

sample_image_path = "tests/sample_face.jpg"

# Load image
image = cv2.imread(sample_image_path)
if image is None:
    print(f"Error: Could not load image at {sample_image_path}")
else:
    # Detect emotion using function 
    result = detect_emotion(image)
    print(f"Detected emotion: {result['dominant_emotion']} ({result['confidence']:.2f})")