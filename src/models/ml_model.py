import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

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
    


    # Evaluation Function
    def evaluate_model(model, X_test, y_test):
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred))
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")

    # Plot Learning Curves
    def plot_learning_curves(history):
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
        plt.title('Learning Curve')
        plt.show()



