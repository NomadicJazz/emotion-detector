from deepface import DeepFace 
import cv2

print("Loading webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam loaded. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Faled to grab frame")
        break

    try:
        # analyze frame with Deepface
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get dominat emotion
        emotion = result[0]['dominant_emotion']

        # Get face coordinates (x,y,w,h)
        region = result[0]['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw emotion text above the box
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2 )

        except Exception as e:
            # Skip frame if detection fails
            pass

        # Show video
        cv2.imshow("Emotion Detector", frame)

        # Press 'Q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()