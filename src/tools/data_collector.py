import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

class DataCollector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, frame):
        frame_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_color)

        if results.pose_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmarks]
            return np.array(landmarks).flatten
        
        return None
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.extract_landmarks(frame)
            if landmarks is not None:
                frames_data.append(landmarks)
        
        cap.release()
        return np.array(frames_data)

    def collect_data(self, data_dir):
        dataset = []
        labels = []

        # exercise_videos/
        #     squats/
        #         video1.mp4
        #         video2.mp4
        #     pushups/
        #         video1.mp4
        #         video2.mp4
        #     ...
        
        for exercise_name in os.listdir(data_dir):
            exercise_path = os.path.join(data_dir, exercise_name)
            if not os.path.isdir(exercise_path):
                continue
                
            print(f"Processing {exercise_name} videos...")
            for video_file in tqdm(os.listdir(exercise_path)):
                if not video_file.endswith(('.mp4', '.avi', '.MOV')):
                    continue
                    
                video_path = os.path.join(exercise_path, video_file)
                video_data = self.process_video(video_path)
                
                if len(video_data) > 0:
                    dataset.append(video_data)
                    labels.append(exercise_name)
        
        return dataset, labels