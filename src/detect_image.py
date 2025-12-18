from deepface import DeepFace
import sys
import cv2

def main():
    if len(sys.argv) != 2:
        print("Usage: python detect_image.py <path_to_image>")
        return

    img_path = sys.argv[1]
    print(f"Analyzing {img_path}...")

    try:
        result = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        print(f"Detected emotion: {emotion}")

        # Show the image with overlay
        img = cv2.imread(img_path)
        cv2.putText(img, f"Emotion: {emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", img)
        cv2.waitKey(0)

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
