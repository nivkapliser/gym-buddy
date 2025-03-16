import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

class DataCollector:
    """
        Class to collect the exercise dataset
        The class uses the MediaPipe Pose model to extract the pose landmarks from the exercise videos
        The pose landmarks are extracted from each frame of the video and stored in a numpy array
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7, # might change to 0.7
            min_tracking_confidence=0.7 # might change to 0.7
        )

        # if dataset already exists, load it and skip the collection process
        # saving the dataset in a csv file after reading it from the video files
        # will be a good idea to avoid the time-consuming process of reading the video files
        

    def extract_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
            Function to extract the pose landmarks from a frame
            Args:
                frame (numpy.ndarray): Input frame
            Returns:
                landmarks (numpy.ndarray): Pose landmarks
        """
        frame_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert to RGB for mediapipe
        results = self.pose.process(frame_color) # process the frame and get the pose landmarks

        # if pose landmarks are detected, extract the landmarks
        if results.pose_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            return np.array(landmarks).flatten()
        
        return None
    
    def process_video(self, video_path: str) -> np.ndarray:
        """
            Function to process a video file and extract the pose landmarks
            Args:
                video_path (str): Path to the video file
            Returns:
                frames_data (numpy.ndarray): Pose landmarks for each frame
        """
        cap = cv2.VideoCapture(video_path) # read the video file
        frames_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.extract_landmarks(frame) # extract pose landmarks from the frame
            if landmarks is not None: 
                frames_data.append(landmarks)
        
        cap.release()
        return np.array(frames_data)

    def collect_dataset(self, data_dir: str) -> tuple:
        """
            Function to collect the exercise dataset
            Args:
                data_dir (str): Path to the directory containing the exercise videos
            Returns:
                dataset (list): List of pose landmarks for each video
                labels (list): List of exercise labels
        """
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