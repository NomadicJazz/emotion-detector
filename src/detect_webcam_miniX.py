import cv2
from fer import FER
from collections import deque, Counter

# --- Configuration ---
N = 10  # number of frames for smoothing
min_face_size = 50  # minimum face width/height in pixels
confidence_threshold = 0.5  # ignore low-confidence predictions
scale = 0.5  # resize frame for performance

# --- Initialize ---
recent_emotions = deque(maxlen=N)
print("Starting webcam...")

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        exit()

    detector = FER(mtcnn=True)  # mini_XCEPTION + MTCNN for face detection
    print("Webcam loaded. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize for faster detection
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        # Detect emotions
        results = detector.detect_emotions(small_frame)

        # Track the largest face only (usually the user)
        if results:
            largest_face = max(results, key=lambda f: f["box"][2] * f["box"][3])
            (x, y, w, h) = largest_face["box"]
            x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)  # scale back

            # Skip tiny faces
            if w < min_face_size or h < min_face_size:
                continue

            emotions = largest_face["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]

            if confidence < confidence_threshold:
                continue  # skip uncertain predictions

            # Add to smoothing deque
            recent_emotions.append(dominant_emotion)
            smooth_emotion = Counter(recent_emotions).most_common(1)[0][0]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{smooth_emotion} ({confidence:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2)

        cv2.imshow("Mini_XCEPTION Emotion Detector", frame)

        # Quit on 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")


