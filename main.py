import cv2
import mediapipe as mp
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Queue to store recent limb positions (e.g., left ankle)
trail_history = deque(maxlen=15)  # Adjust maxlen for trail length

cap = cv2.VideoCapture(2)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5, model_complexity=2) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, mark the image as not writable to pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw pose annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Draw the main landmarks and connections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # trail right wrist
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            trail_history.append(right_wrist)
            mp_drawing.draw_landmarks(
                image,
                trail_history,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )


        # Flip the image horizontally for a selfie-view display
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()