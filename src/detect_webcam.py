from deepface import DeepFace
import cv2
from collections import deque, Counter

print("Starting webcam...")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Keep last N emotions to smooth predictions
N = 5
recent_emotions = deque(maxlen=N)

print("Webcam loaded. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    try:
        # Use MTCNN for better face detection
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            detector_backend='mtcnn',
            enforce_detection=False
        )

        # Get dominant emotion and probability
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]

        # Add to deque for smoothing
        recent_emotions.append(emotion)
        # Most common emotion in last N frames
        smooth_emotion = Counter(recent_emotions).most_common(1)[0][0]

        # Draw bounding box
        region = result[0]['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show emotion + confidence
        cv2.putText(frame,
                    f"{smooth_emotion} ({confidence:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2)

    except Exception as e:
        # Skip frame if detection fails
        pass

    # Display
    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

