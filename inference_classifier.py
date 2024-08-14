import pickle  # Import the pickle module to load the saved model
import cv2  # Import OpenCV for image and video processing
import mediapipe as mp  # Import MediaPipe for hand landmark detection
import numpy as np  # Import NumPy for array manipulation

# Load the previously trained gesture recognition model from a pickle file
model_dict = pickle.load(open('./model.p', 'rb'))  # Load the model dictionary
model = model_dict['model']  # Extract the model from the dictionary

# Initialize video capture to use the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands module for detecting hand landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure the Hands object for real-time processing
hands = mp_hands.Hands(static_image_mode=False,  # Set to False to enable continuous detection in video streams
                       max_num_hands=2,  # Detect up to 2 hands
                       min_detection_confidence=0.3)  # Minimum confidence threshold for detection

# Dictionary to map prediction outputs to corresponding alphabet letters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z', 26: '-'}

# Start the video capture loop
while True:
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        break  # If the frame isn't captured properly, exit the loop

    H, W, _ = frame.shape  # Get the height and width of the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB for Mediapipe
    results = hands.process(frame_rgb)  # Process the frame to detect hand landmarks

    if results.multi_hand_landmarks:  # If hand landmarks are detected
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label  # Determine if the hand is Left or Right

            if hand_label == 'Left':  # Only process the left hand
                data_aux = []  # Initialize a list to hold normalized landmark coordinates
                x_ = []  # List to store x-coordinates of landmarks
                y_ = []  # List to store y-coordinates of landmarks

                # Draw the hand landmarks and connections on the frame
                mp_drawing.draw_landmarks(frame,  # The frame to draw on
                                          hand_landmarks,  # The detected hand landmarks
                                          mp_hands.HAND_CONNECTIONS,  # Connections between landmarks
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                # Collect the x and y coordinates of each landmark
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    x_.append(x)
                    y_.append(y)

                # Normalize the coordinates relative to the top-left corner of the bounding box around the hand
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))  # Normalize x-coordinates
                    data_aux.append(landmark.y - min(y_))  # Normalize y-coordinates

                # Calculate the bounding box for the hand
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Predict the character using the trained model
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]  # Map the prediction to a character

                # Draw the bounding box and the predicted character on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
            else:
                print("Left hand detected")
                x_ = [landmark.x for landmark in hand_landmarks.landmark]  # List comprehension for x-coordinates
                y_ = [landmark.y for landmark in hand_landmarks.landmark]  # List comprehension for y-coordinates

                # Calculate the bounding box for the right hand
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Draw the bounding box and label it as "Save"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, "Save", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

    # Display the frame with the drawings and predictions
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
