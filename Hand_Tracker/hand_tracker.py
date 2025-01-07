import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize the HandLandmarker
base_options = python.BaseOptions(model_asset_path='Hand_Tracker\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Create a VideoCapture object to read from webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Loop until 'q' is pressed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for better performance
    frame = cv2.resize(frame, (540, 380), interpolation=cv2.INTER_CUBIC)

    # Convert the frame to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Perform hand detection
    detection_result = detector.detect(mp_image)

    # Draw landmarks on the frame
    for hand_landmarks in detection_result.hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    # Display the resulting frame
    cv2.imshow('Hand Detection', frame)

    # Exit the loop on pressing 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
