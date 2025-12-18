from fer import FER
import cv2

# Initialize the detector globally so it can be reused
detector = FER(mtcnn=True)

def detect_emotion(image):
    """
    Detects the dominant emotion in a given image using FER (mini_XCEPTION + MTCNN).

    Args:
        image (numpy.ndarray): Input image (BGR format).

    Returns:
        dict: {
            "dominant_emotion": str,
            "confidence": float
        }
        Returns {"dominant_emotion": None, "confidence": 0.0} if no face detected or error occurs.
    """
    try:
        results = detector.detect_emotions(image)
        if results:
            emotions = results[0]["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]
            return {"dominant_emotion": dominant_emotion, "confidence": confidence}
    except Exception as e:
        print(f"[ERROR] Failed to detect emotions: {e}")

    # Default if no face detected or exception occurred
    return {"dominant_emotion": None, "confidence": 0.0}


if __name__ == "__main__":
    sample_image_path = "tests/sample_face.jpg"  # Path to a test image

    try:
        # Load the image
        image = cv2.imread(sample_image_path)
        if image is None:
            raise ValueError(f"Could not load image at {sample_image_path}")

        # Detect emotion
        result = detect_emotion(image)
        dominant = result["dominant_emotion"]
        confidence = result["confidence"]
        print(f"Detected emotion: {dominant} ({confidence:.2f})")

        # Show the image with overlay if a face was detected
        if dominant:
            cv2.putText(
                image,
                f"Emotion: {dominant} ({confidence:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow("Emotion Detection", image)
            print("Press any key on the image window to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No face detected in the image.")

    except Exception as e:
        print(f"[ERROR] {e}")
