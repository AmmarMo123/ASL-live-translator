import os  # Import the os module for file and directory operations
import cv2  # Import the OpenCV library for computer vision tasks

# Define the directory where the data will be stored
DATA_DIR = './data'

# Check if the data directory exists, if not, create it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes you want to collect data for
number_of_classes = 27

# Define the number of images to collect for each class
dataset_size = 100

# Start capturing video from the webcam (device index 0)
cap = cv2.VideoCapture(0)

# Loop through each class to collect data
for j in range(number_of_classes):
    # Create a directory for each class if it doesn't already exist
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    # Wait for user input to start data collection
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        # Display a message on the frame instructing the user to press 'Q' to start
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Show the frame with the message

        # If the user presses 'Q', break out of the loop and start data collection
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0  # Initialize a counter for the number of images collected

    # Collect the specified number of images for the current class
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a frame from the webcam
        cv2.imshow('frame', frame)  # Display the current frame
        cv2.waitKey(25)  # Wait for a short period to control the frame rate

        # Save the captured frame as an image file in the class directory
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1  # Increment the counter

# Release the webcam resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
