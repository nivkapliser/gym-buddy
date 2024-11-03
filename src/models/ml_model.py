import tensorflow as tf
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from  .data_models import Exercise

class MLModel:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(99,)), # input_shape: 33 landmarks * 3 coordinates (x, y, z)
            tf.keras.layers.Dropout(0.3), # for overfitting
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax') # number of neurons = num of exercises, softmax for multimodal classification
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def preprocess_landmarks(self, landmarks) -> np.ndarray:
        """Convert landmarks to flat array of coordinates"""
        coordinates = []
        for landmark in landmarks:
            coordinates.extend([landmark.x, landmark.y, landmark.z])
        return np.array(coordinates)
    
    def train(self, landmarks_data: List[np.ndarray], labels: List[Exercise]): # labels is a list of exercises name
        """Train the model on the landmark data"""
        X = landmarks_data
        X_scaled = self.scaler.fit_transform(X)

        # One-hot encoding on labels
        y = tf.keras.utils.to_categorical([ex.value for ex in labels])

        self.model.fit(X_scaled, y, epoch=50, validation_split=0.2)
        
    def predict(self, landmarks) -> Tuple[Exercise, float]:
        """Predict exercise from landmarks"""
        X = self.preprocess_landmarks(landmarks)
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        
        predictions = self.model.predict(X_scaled)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        
        return Exercise(predicted_idx), confidence