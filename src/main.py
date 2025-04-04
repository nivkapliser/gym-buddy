from tools.data_collector import DataCollector
from tools.data_preprocessing import DataPreprocessor, LabelEncoder
from models.ml_model import ExerciseModel
from exercise_detector import ExerciseDetector
import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence = 0.7, min_tracking_confidence = 0.7)

SEQUENCE_LENGTH = 30

def train_model(data_dir: str) -> tuple:
    """
        Function to train the exercise recognition model
        Args:
            data_dir (str): Path to the directory containing the exercise videos
        Returns:
            model (tf.keras.Model): Trained exercise recognition model
            label_encoder (LabelEncoder): Label encoder object
    """
    # Collect data
    collector = DataCollector()
    data, labels = collector.collect_dataset(data_dir)
    
    # Preprocess data for training
    preprocessor = DataPreprocessor(sequence_length=SEQUENCE_LENGTH)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(data, labels)
    
    # find the number of features and classes
    n_features = X_train.shape[2]
    n_classes = len(np.unique(y_train))
    
    # Train the model
    exercise_model = ExerciseModel(n_classes, sequence_length=SEQUENCE_LENGTH, n_features=n_features)
    model, history = exercise_model.train(X_train, y_train, X_test, y_test) 
    
    # Save model and label encoder
    model.save('exercise_model.h5')
    np.save('label_encoder.npy', preprocessor.label_encoder.classes_)

    # Evaluate the model
    exercise_model.evaluate_model(model, X_test, y_test)
        
    # Plot learning curves
    exercise_model.plot_learning_curves(history)
    
    return model, preprocessor.label_encoder

def run_real_time_detection():
    """
        Function to run real-time exercise detection
        it detects the exercise being performed by the user in real-time using the webcam
        and displays the exercise name and confidence on the screen
    """
    # Load model and label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_encoder.npy')
    detector = ExerciseDetector('exercise_model.h5', label_encoder)

    cap = cv2.VideoCapture(0)
    # frame_skip_rate = 3# Skip every 3 frames to speed up the process
    # frame_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame_counter += 1
        # if frame_counter % frame_skip_rate != 0:
        #     cv2.imshow('Exercise Detection', frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #     continue

        frame = cv2.flip(frame, 1)

        # Print the pose landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(102, 102, 255), thickness=3, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(157, 47, 165), thickness=3, circle_radius=2))
            
        exercise_name, confidence = detector.detect_exercise(frame)
        
        if exercise_name:
            cv2.putText(
                frame,
                f"{exercise_name} ({confidence:.2f})",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 0),
                4
            )
            # Draw pose landmarks
            
        cv2.imshow('Exercise Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # First time: train the model
    #model, label_encoder = train_model("data\exercise_videos")
    
    # Run real-time detection
    run_real_time_detection()