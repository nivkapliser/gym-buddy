import cv2
import mediapipe as mp
from src.models.data_models import Exercise
from src.tools.angleHelper import calculate_angle, get_left_knee_angle, get_right_knee_angle
from collections import deque
import numpy as np

window_name = "Squat Detection"

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence = 0.7, min_tracking_confidence = 0.7)

angle_history = deque(maxlen=5)

# Initialize OpenCV for capturing video
# cap = cv2.VideoCapture(0)  # Set to 0 for webcam
cap = cv2.VideoCapture('data\exercise_videos\deadlift\deadlift_23.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_knee_angle = get_left_knee_angle(landmarks)
        right_knee_angle = get_right_knee_angle(landmarks)

        # Add the current angle to the history
        angle_history.append((left_knee_angle, right_knee_angle))

        # Calculate the average angle for smoothing
        avg_left_knee_angle = np.mean([angle[0] for angle in angle_history])
        avg_right_knee_angle = np.mean([angle[1] for angle in angle_history])

        knee_left_coords = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * frame.shape[1]),
                            int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * frame.shape[0]))
        knee_right_coords = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * frame.shape[0]))



        # Show exercise name
        cv2.putText(frame, 'Exercise', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, Exercise.SQUAT.value, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        # left knee angle
        cv2.putText(frame,str(round(avg_left_knee_angle)), knee_left_coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # right knee angle
        cv2.putText(frame, str(round(avg_right_knee_angle)), knee_right_coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(102, 102, 255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(157, 47, 165), thickness=2, circle_radius=2))

    cv2.imshow(window_name, frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
