# gym-buddy - Real Time Exercise Recognition

This project implements real-time exercise recognition using a custom-built machine learning pipeline. The application leverages MediaPipe for pose estimation and TensorFlow for model training, focusing on exercises like squats, push-ups, dips, and pull ups.

## Project Structure

* `data_collector.py` – Captures exercise data from webcam or video inputs.

* `data_preprocessing.py` – Handles data cleaning, normalization, and augmentation.

* `ml_model.py` – Contains the LSTM model, training pipeline, and evaluation functions.

* `exercise_detector.py` – Real-time exercise detection script.

* `main.py` – Debugging script to test the full recognition pipeline.

## Model Architecture

The model leverages a LSTM to analyze temporal patterns in exercise movements, enabling highly accurate classifications. The architecture consists of:

* **Two LSTM Layers:** Captures forward temporal dependencies.

* **Dropout Layers:** Reduces overfitting by randomly deactivating neurons.

* **Dense Layers:** Employs ReLU activation for non-linear feature extraction.

* **Input:** 30-frame sequences of body landmarks (x,y,z coordinates).

## Data

* **Custom Dataset:** Collected through data_collector.py using webcam footage.

* **Workout/Exercises Video from Kaggle:** Multiple exercises vides taken mainly from Youtube.

* **Train-Validation-Test Split:** 70% training, 15% validation, 15% test.

## Installation
```
# Clone the repository
git clone https://github.com/nivkapliser/gym-buddy.git
cd exercise-recognizer

# Install dependencies
pip install -r requirements.txt
```
## Usage
```
# Collect exercise data
python data_collector.py

# Preprocess data
python data_preprocessing.py

# Train the model
python ml_model.py

# Run real-time exercise detection
python exercise_detector.py
```
## Model Training and Evaluation

* **Training:** Model training occurs in ml_model.py, with the best models saved to the models/ directory.

* **Evaluation:**

  * Generates classification reports and confusion matrices.

  * Learning curves (accuracy and loss) are plotted after training.

## Results

* **Accuracy:** Consistently achieves high accuracy (>97%) on diverse test sets.

* **Robustness:** Performs reliably across varying environments and angles.

## Visualization

* **Confusion Matrices:** Visualize classification performance for each exercise class.

* **Learning Curves:** Plots of accuracy and loss during training.

* **Pose Visualization:** Real-time pose landmark visualization during detection.

## Future Improvements

* **Additional Exercises:** Expand to include more complex and diverse exercises.

* **Add Web/Mobile App:** for better users experience.

* **Form Feedback:** Real-time feedback to improve exercise form and reduce injury risk.

## Acknowledgments

* MediaPipe for pose estimation

* TensorFlow/Keras for model development

* Community datasets and synthetic data augmentation for training
