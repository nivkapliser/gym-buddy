import cv2
import mediapipe as mp
import numpy as np
from ..models.data_models import Exercise
from pathlib import Path
from datetime import datetime

class DataCollector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.data_dir = Path('data/training')
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect_data(self, exercise: Exercise, num_samples: int = 100):
        samples = []
        cap = cv2.VideoCapture(0)

        sample_count = 0
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            if results.pose_landmarks:
                self.draw_landmarks(frame, results)

            cv2.putText(frame, f"Samples: {sample_count}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and results.pose_landmarks:
                # Save landmark data
                landmarks_data = self.extract_landmarks(results.pose_landmarks)
                samples.append({
                    'landmarks': landmarks_data,
                    'exercise': exercise.value,
                    'timestamp': datetime.now().isoformat()
                })
                sample_count += 1
                print(f"Captured sample {sample_count}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected data
        self.save_samples(samples, exercise)