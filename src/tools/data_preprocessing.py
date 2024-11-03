import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.label_encoder = LabelEncoder()
        
    def create_sequences(self, data, labels):
        sequences = []
        sequence_labels = []
        
        for video_data, label in zip(data, labels):
            for i in range(0, len(video_data) - self.sequence_length + 1):
                seq = video_data[i:i + self.sequence_length]
                sequences.append(seq)
                sequence_labels.append(label)
                
        return np.array(sequences), np.array(sequence_labels)
    
    def prepare_data(self, data, labels):
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Create sequences
        X, y = self.create_sequences(data, encoded_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test