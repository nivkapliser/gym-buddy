import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
        Class to preprocess the exercise dataset
        The class preprocesses the data for training the exercise recognition model
        The class creates sequences of pose landmarks from the video data
        The sequences are created with a fixed length (1sec video) and the labels are encoded using the LabelEncoder
    """
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.label_encoder = LabelEncoder()
        
    def create_sequences(self, data: list[np.ndarray], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
            Function to create sequences of pose landmarks from the video data
            Args:
                data (list): List of pose landmarks for each video
                labels (list): List of labels for each video
            Returns:
                sequences (numpy.ndarray): Array of sequences
                sequence_labels (numpy.ndarray): Array of sequence labels
        """
        sequences = []
        sequence_labels = []
        
        for video_data, label in zip(data, labels):
            for i in range(0, len(video_data) - self.sequence_length + 1):
                seq = video_data[i:i + self.sequence_length]
                sequences.append(seq)
                sequence_labels.append(label)
                
        return np.array(sequences), np.array(sequence_labels)
    
    def prepare_data(self, data: list[np.ndarray], labels: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            Function to preprocess the exercise dataset
            splits the data into training and testing sets
            Args:
                data (list): List of pose landmarks for each video
                labels (list): List of labels for each video
            Returns:
                X_train (numpy.ndarray): Training data
                X_test (numpy.ndarray): Testing data
                y_train (numpy.ndarray): Training labels
                y_test (numpy.ndarray): Testing labels
        """
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Create sequences
        X, y = self.create_sequences(data, encoded_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test