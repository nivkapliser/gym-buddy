from tools.data_collector import DataCollector
from tools.data_preprocessing import DataPreprocessor, LabelEncoder
from models.ml_model import ExerciseModel
from exercise_detector import ExerciseDetector
import cv2
import numpy as np

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
            
        exercise_name, confidence = detector.detect_exercise(frame)
        
        if exercise_name:
            cv2.putText(
                frame,
                f"{exercise_name} ({confidence:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
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