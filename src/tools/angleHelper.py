import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle between three points a, b, and c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # Calculate the cosine of the angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # Convert the angle from radians to degrees
    angle = np.degrees(angle)

    if angle > 180.0:
        angle = 360 - angle

    return angle
def get_left_knee_angle(landmarks):
    if not landmarks:
        return 0
    if landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility < 0.1 or landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility < 0.1 or landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].visibility < 0.1:
        return 0

    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP].z]
    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z]
    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z]
    return calculate_angle(hip_left, knee_left, ankle_left)

def get_right_knee_angle(landmarks):
    if not landmarks:
        return 0
    if landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility < 0.1 or landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].visibility < 0.1 or landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility < 0.1:
        return 0

    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z]
    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z]
    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].z]
    return calculate_angle(hip_right, knee_right, ankle_right)