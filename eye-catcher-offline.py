import cv2
import numpy as np
import dlib
import csv
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True, help='path to facial landmark predictor')
ap.add_argument('-v', '--video', type=str, default="", help='path to input video file')
args = vars(ap.parse_args())


# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load video file
video_file = (args['video'])
cap = cv2.VideoCapture(video_file)

# Initialize variables
blink_counter = 0
blink_flag = False
blink_timestamps = []

# Loop through frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract left and right eye landmarks
        left_eye = []
        right_eye = []
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        # Calculate eye aspect ratio (EAR)
        left_ear = eye_aspect_ratio(np.array(left_eye))
        right_ear = eye_aspect_ratio(np.array(right_eye))

        # Average EAR of both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check for blink
        if ear < 0.2:
            if not blink_flag:
                blink_flag = True
                blink_counter += 1
                blink_timestamps.append((cap.get(cv2.CAP_PROP_POS_MSEC), cap.get(cv2.CAP_PROP_POS_FRAMES)))
        else:
            blink_flag = False

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()

# Save blink timestamps to CSV file
csv_file = "blink_timestamps.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp (ms)', 'Frame Number'])
    for timestamp in blink_timestamps:
        writer.writerow(timestamp)

print("Blink detection complete. Blink timestamps saved to", csv_file)
