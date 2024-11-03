import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

class ExerciseModel:
    def __init__(self, n_classes, sequence_length, n_features):
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.n_features = n_features
        
    def build_model(self):
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers
        x = LSTM(128, return_sequences=True)(input_layer)
        x = Dropout(0.3)(x)
        x = LSTM(64)(x)
        x = Dropout(0.3)(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        output_layer = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        model = self.build_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return model, history





# import tensorflow as tf
# import numpy as np
# from typing import List, Tuple
# from sklearn.preprocessing import StandardScaler
# from  .data_models import Exercise

# class MLModel:
#     def __init__(self):
#         self.model = self._build_model()
#         self.scaler = StandardScaler()

#     def _build_model(self):
#         model = tf.keras.Sequential([
#             tf.keras.layers.Dense(128, activation='relu', input_shape=(99,)), # input_shape: 33 landmarks * 3 coordinates (x, y, z)
#             tf.keras.layers.Dropout(0.3), # for overfitting
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(32, activation='relu'),
#             tf.keras.layers.Dense(3, activation='softmax') # number of neurons = num of exercises, softmax for multimodal classification
#         ])

#         model.compile(optimizer='adam',
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])
        
#         return model

#     def preprocess_landmarks(self, landmarks) -> np.ndarray:
#         """Convert landmarks to flat array of coordinates"""
#         coordinates = []
#         for landmark in landmarks:
#             coordinates.extend([landmark.x, landmark.y, landmark.z])
#         return np.array(coordinates)
    
#     def train(self, landmarks_data: List[np.ndarray], labels: List[Exercise]): # labels is a list of exercises name
#         """Train the model on the landmark data"""
#         X = landmarks_data
#         X_scaled = self.scaler.fit_transform(X)

#         # One-hot encoding on labels
#         y = tf.keras.utils.to_categorical([ex.value for ex in labels])

#         self.model.fit(X_scaled, y, epoch=50, validation_split=0.2)
        
#     def predict(self, landmarks) -> Tuple[Exercise, float]:
#         """Predict exercise from landmarks"""
#         X = self.preprocess_landmarks(landmarks)
#         X_scaled = self.scaler.transform(X.reshape(1, -1))
        
#         predictions = self.model.predict(X_scaled)[0]
#         predicted_idx = np.argmax(predictions)
#         confidence = predictions[predicted_idx]
        
#         return Exercise(predicted_idx), confidence