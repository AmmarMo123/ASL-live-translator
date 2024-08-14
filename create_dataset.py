import os 
import pickle  # Import the pickle module for serializing and saving data
import mediapipe as mp  # Import MediaPipe, a library for machine learning and computer vision tasks
import cv2

# Initialize MediaPipe's hand detection and drawing objects
mp_hands = mp.solutions.hands  # Load the hand detection solution from MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing landmarks and connections on images
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined drawing styles for landmarks and connections

# Create a hand detection object with specific settings
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory where the image data is stored
DATA_DIR = './data'

# Initialize lists to hold the processed data and corresponding labels (landmark coordinates)
data = []
labels = []

# Iterate through each directory in the data folder (each directory corresponds to a class/alphabet)
for dir_ in os.listdir(DATA_DIR):
    # Iterate through each image in the class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # List to hold the hand landmark data for the current image

        x_ = []  # List to hold the x-coordinates of hand landmarks
        y_ = []  # List to hold the y-coordinates of hand landmarks

        # Read the image using OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Convert the image from BGR (OpenCV default) to RGB (required by MediaPipe)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # If hand landmarks are detected, iterate over each hand detected in the image
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect all the x and y coordinates of the hand landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the coordinates by subtracting the minimum x and y values
                # This ensures that the landmarks are relative to the top-left corner of the bounding box around the hand
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Append normalized x-coordinate
                    data_aux.append(y - min(y_))  # Append normalized y-coordinate

            # Append the processed landmark data and corresponding label to the data and labels lists
            data.append(data_aux)
            labels.append(dir_)

# Save the data and labels lists into a pickle file for later use
f = open('data.pickle', 'wb')  # Open a file in write-binary mode
pickle.dump({'data': data, 'labels': labels}, f)  # Save the data and labels to the file
f.close()  # Close the file
