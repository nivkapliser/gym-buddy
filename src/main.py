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

def train_model(data_dir):
    # Collect data
    collector = DataCollector()
    data, labels = collector.collect_dataset(data_dir)
    
    # Preprocess data
    preprocessor = DataPreprocessor(sequence_length=30)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(data, labels)
    
    # Train model
    n_features = X_train.shape[2]
    n_classes = len(np.unique(y_train))
    
    exercise_model = ExerciseModel(n_classes, sequence_length=30, n_features=n_features)
    model, history = exercise_model.train(X_train, y_train, X_test, y_test)
    
    # Save model and label encoder
    model.save('exercise_model.h5')
        
    # Evaluate the model
    exercise_model.evaluate_model(model, X_test, y_test)
        
    # Plot learning curves
    exercise_model.plot_learning_curves(history)

    np.save('label_encoder.npy', preprocessor.label_encoder.classes_)
    
    return model, preprocessor.label_encoder

def run_real_time_detection():
    # Load model and label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_encoder.npy')
    
    detector = ExerciseDetector('exercise_model.h5', label_encoder)
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
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