from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pickle
import mediapipe as mp

app = Flask(__name__)  # Initialize Flask app
CORS(app)  # Enable CORS

# Load the gesture recognition model and other required components
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe components for hand tracking and drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '_'}

def process_frame(frame_data):
    # Decode base64 image data
    img_data = base64.b64decode(frame_data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the frame from BGR (OpenCV default) to RGB (used by Mediapipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    H, W, _ = frame.shape
    predicted_character = None
    left_hand_present = False

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label

            if hand_label == 'Left':  
                data_aux = []
                x_ = []
                y_ = []

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame_rgb,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Collect x and y coordinates of hand landmarks
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    x_.append(x)
                    y_.append(y)

                # Normalize the coordinates and store them in data_aux
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                # Define a bounding box around the hand
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Use the model to predict the gesture based on landmarks
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Draw the bounding box and predicted character on the frame
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame_rgb, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
            else:  
                left_hand_present = True
                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]

                # Draw bounding box with the label "Save"
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame_rgb, "Save", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

    # Convert the frame back to BGR before encoding
    processed_frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Encode the processed frame to base64
    retval, buffer = cv2.imencode('.jpg', processed_frame_bgr)
    img_str = base64.b64encode(buffer).decode()

    return img_str, predicted_character, left_hand_present

@app.route('/')
def index():
    return "Flask server is running."

@app.route('/video', methods=['POST'])
def video():
    # Get the base64-encoded image data from the POST request
    frame_data = request.form['image']
    
    # Process the frame
    processed_frame, predicted_character, left_hand_present = process_frame(frame_data)

    # Send the processed frame, predicted character, and left hand presence back to the client
    return jsonify({'processed_frame': processed_frame, 'predicted_character': predicted_character, 'left_hand_present': left_hand_present})

if __name__ == "__main__":
    app.run(debug=True, port=5002)
