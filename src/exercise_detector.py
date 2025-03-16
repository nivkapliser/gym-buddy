import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from typing import Optional

class ExerciseDetector:
    """
        Class to detect the exercise being performed by the user
    """
    def __init__(self, model_path: str, label_encoder , sequence_length: int = 30):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.model = tf.keras.models.load_model(model_path)
        self.label_encoder = label_encoder
        self.sequence_length = sequence_length
        self.sequence_buffer = []
        
    def preprocess_frame(self, frame: np.array) -> Optional[np.array]:
        """
            Function to preprocess the frame for pose detection
            Args:
                frame (np.array): Frame from the webcam
            Returns:
                landmarks (np.array): Pose landmarks detected in the frame
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            return np.array(landmarks).flatten()
        return None
    
    def detect_exercise(self, frame: np.array) -> tuple[Optional[str], float]:
        """
            Function to detect the exercise being performed by the user
            Args:
                frame (np.array): Frame from the webcam
            Returns:
                exercise_name (str): Name of the exercise being performed
                confidence (float): Confidence of the model
        """
        landmarks = self.preprocess_frame(frame)
        
        if landmarks is not None:
            self.sequence_buffer.append(landmarks)
            
            if len(self.sequence_buffer) > self.sequence_length:
                self.sequence_buffer.pop(0)
            
            if len(self.sequence_buffer) == self.sequence_length:
                sequence = np.array([self.sequence_buffer])
                prediction = self.model.predict(sequence, verbose=0)[0]
                exercise_id = np.argmax(prediction)
                confidence = prediction[exercise_id]
                
                if confidence > 0.9:  # Confidence threshold
                    exercise_name = self.label_encoder.inverse_transform([exercise_id])[0]
                    return exercise_name, confidence
                
        return None, 0.0