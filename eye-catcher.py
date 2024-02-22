import cv2
import dlib
from scipy.spatial import distance

def calculate_eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark
    C = distance.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Load the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the eye landmarks indices
left_eye_indices = [36, 37, 38, 39, 40, 41]
right_eye_indices = [42, 43, 44, 45, 46, 47]

# Initialize variables
blink_counter = 0
blink_duration = 0
is_blinking = False

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Detect the facial landmarks
        landmarks = predictor(gray, face)

        # Extract the left and right eye landmarks
        left_eye = [(landmarks.part(index).x, landmarks.part(index).y) for index in left_eye_indices]
        right_eye = [(landmarks.part(index).x, landmarks.part(index).y) for index in right_eye_indices]

        # Calculate the eye aspect ratio for both eyes
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)

        # Average the eye aspect ratio for both eyes
        avg_ear = (left_ear + right_ear) / 2.0

        # Check if the person is blinking
        if avg_ear < 0.2:
            if not is_blinking:
                is_blinking = True
                blink_counter += 1
        else:
            is_blinking = False

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the number of blinks and their duration
print("Number of blinks:", blink_counter)
print("Blink duration:", blink_duration)