# Purpose: Draw lines between consecutive body part positions to visualize movement patterns

import cv2
import mediapipe as mp
import numpy as np
import csv
import random

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Start capturing video
cap = cv2.VideoCapture(0)

# Create a blank image to store movement lines
movement_image = np.zeros((480, 640, 3), dtype=np.uint8)

# Define previous landmark positions (initially None)
prev_landmarks = None
prev_head_position = None

# Random colors for each landmark -> unite
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(33)]
head_color = (255, 255, 0)  # Yellow for head path

# Open a CSV file to store the coordinates
with open('movement_coordinates.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'Landmark_Index', 'X', 'Y', 'Z'])  # CSV header

    frame_count = 0  # To keep track of frame numbers

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose detection
        results = pose.process(rgb_frame)

        # If landmarks are detected, process them
        if results.pose_landmarks:
            # Extract landmark positions
            landmarks = results.pose_landmarks.landmark

            # Calculate the head position as the average of specific landmarks --> make helper function
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

            head_x = (nose.x + left_eye.x + right_eye.x + left_ear.x + right_ear.x) / 5
            head_y = (nose.y + left_eye.y + right_eye.y + left_ear.y + right_ear.y) / 5
            head_z = (nose.z + left_eye.z + right_eye.z + left_ear.z + right_ear.z) / 5

            curr_head_position = (int(head_x * 640), int(head_y * 480))

            # Write head coordinates to the CSV file
            writer.writerow([frame_count, "head", head_x, head_y, head_z])

            # Draw line for head movement if previous position exists
            if prev_head_position:
                cv2.line(movement_image, prev_head_position, curr_head_position, head_color, 2)

            # Update the previous head position
            prev_head_position = curr_head_position

            landmarks = landmarks[11:]
            # Draw lines for other body parts
            if prev_landmarks:
                for i, landmark in enumerate(landmarks):
                    # if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    #     continue

                    curr_x, curr_y = int(landmark.x * 640), int(landmark.y * 480)
                    prev_x, prev_y = int(prev_landmarks[i].x * 640), int(prev_landmarks[i].y * 480)

                    # Draw line between previous and current positions for each body part
                    cv2.line(movement_image, (prev_x, prev_y), (curr_x, curr_y), colors[i], 2)

                    # Write coordinates to the CSV file
                    writer.writerow([frame_count, i, landmark.x, landmark.y, landmark.z])

            # Update previous landmarks
            prev_landmarks = landmarks

        # Show the current movement lines overlay
        overlay = cv2.addWeighted(frame, 0.5, movement_image, 0.5, 0)
        cv2.imshow('Body Movement Tracking', overlay)

        # Increment frame count
        frame_count += 1

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Save the movement image
cv2.imwrite('body_movement_lines.png', movement_image)

# Release resources
cap.release()
cv2.destroyAllWindows()
