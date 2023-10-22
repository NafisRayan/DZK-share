import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing_face = mp.solutions.drawing_utils

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing_hands = mp.solutions.drawing_utils

# Initialize MediaPipe Face Mesh Detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_mesh = mp.solutions.drawing_utils

# Initialize the webcam or load an image
# For an image, replace cv2.VideoCapture(0) with the image file path
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection, mp_hands.Hands() as hands, mp_face_mesh.FaceMesh() as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results_face = face_detection.process(image_rgb)

        # Perform hand detection
        results_hands = hands.process(image_rgb)

        # Perform face mesh detection
        results_mesh = face_mesh.process(image_rgb)

        if results_face.detections:
            for detection in results_face.detections:
                mp_drawing_face.draw_detection(image, detection)

        if results_hands.multi_hand_landmarks:
            for landmarks in results_hands.multi_hand_landmarks:
                mp_drawing_hands.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

        if results_mesh.multi_face_landmarks:
            for landmarks in results_mesh.multi_face_landmarks:
                mp_drawing_mesh.draw_landmarks(image, landmarks, mp_face_mesh.FACE_CONNECTIONS)

        cv2.imshow('Face, Hand, and Mesh Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()