This is a Sign Language to English text interpreter, which uses openCV and MediaPipe to detect hand motion, and a  pre-trained random forest classifier model to predict common ASL alphabets. A frontend React web app and a backend Python Flask server were developed to extend the project and make it accessible on the web.

## Process to train AI model that predicts alphabets.
**Create dataset** -> **Extract desired information from data** -> **Train classifier** -> **Test model**

### Step 1:
Run `collect_imgs.py`, collect photos for all the ASL alphabets. This script will generate a ./data directory with classes, encoded as number, and each number will contain all the photos taken for the given alphabet.

### Step 2:
Extract the position of the hand by converting them to landmarks using MediaPipe, meaning we will be converting each image of the hands into a collection of points that determine the position of the hand. This reduces our dataset from a whole images to an array of around 20 points, while maintaining the information we need. This is done using `create_dataset.py`.

### Step 3:
Train classifier with `train_classifier.py` using the MediaPipe points with the random forest classifier from scikit learn. When testing, the model was able to predict the test data with an accuracy score of 100%. The model is then save in the file “model.p” using pickle.

### Step 4:
Test the model with real time webcam feed using the `inference_class.py` script. The script looks for hand land marks in each frame, and the coordinates are used as input to the model, which predicts the corresponding letter. The predicted character, landmarks and a bounding box are drawn on the frame.

> Thank you to https://github.com/computervisioneng/sign-language-detector-python for providing the concepts and code related to creating the model and using openCV and MediaPipe.

# Supporting the translator on the web:
## Goal: Make the sign language translation available on the web, so it’s easily accessible and available to everyone.
This project extends the work done by Computer vision engineer to:
- Create a ***web application*** to capture and process frames for letter prediction.
    - **Front end**
        - Access users webcam through the browser using `getUserMedia` API.
        - Send video frames to a Flask server for gesture recognition.
        - Updates the UI based on the returned Frame, predicted character, and if the left hand is present.
            - Show the predicted character.
            - Append to a string if the left hand is detected.
        - Other misc functionalities (clearing string, start/stop processing).
    - **Back end**
        - Receives video frames from the frontend/client.
        - Detect hand landmarks from video frames using MediaPipe.
        - Predicts the gesture using the pre-trained model.
        - Detects if the left hand is present.
        - Draws the predicted character, hand landmarks and a bounding box on the hands.
        - Returns the processed frame, predicted character, and if the left hand is present in the frame.


## Demo:
https://github.com/user-attachments/assets/8d9edc17-4c03-4e24-aa20-19778899b471

### Try out the project!
Clone the repo:
```
git clone https://github.com/AmmarMo123/ASL-live-translator.git
```

Install dependencies:
```
pip3 install requirements.txt
```

Run flask backend:
```
python3 app.py
```

Set up React Application, navigate to react directory:
```
cd my-app
```

Install dependencies:
```
npm install
```

Start the development server:
```
npm start
```

## Deploying notes
- Vercel: the OpenCV and SciKit learn libraries are too large, and take up too much space for the free Vercel plan.
- AWS: Deployed using EC2 instances.
    - Used Nginx to act as a frontend reverse proxy to handle client requests and responses. For dynamic content, it forwards the requests to Gunicorn.
    - Used Gunicorn to interface between flask python application and web server. It receives the requests from Nginx, processes them using the Python web application, and returns the responses back to Nginx, which then sends them to the client.
    - Obtained a free domain, and obtained an SSL certificate from “Let's Encrypt" so that the deployed application can access the camera through the browser.
    - Issues
        - Application was pretty slow - probably because I was using AWS free tier, and the application would send 30 frames per second back and forth.
        - Got invoices, so cancelled EC2 server.

## Todo:
- Retry deploying with AWS, and see why a lot of storage was taken and if it can be optimised.
- Explore other free options for deployment.
