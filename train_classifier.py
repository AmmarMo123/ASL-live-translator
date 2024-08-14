import pickle
from sklearn.ensemble import RandomForestClassifier  # Import the RandomForestClassifier from scikit-learn
from sklearn.model_selection import train_test_split  # Import function to split data into training and testing sets
from sklearn.metrics import accuracy_score  # Import function to calculate the accuracy of the model
import numpy as np  # Import NumPy for array manipulation

# Load the dataset from the previously saved pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert the loaded data and labels to NumPy arrays to allow for processing usinf scikit
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets, with 20% of the data used for testing, and 80% for training
# shuffle=True randomizes the data before splitting
# stratify=labels ensures that the training and testing sets have the same proportion of each class
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize a RandomForestClassifier model
model = RandomForestClassifier()

# Train the model using the training data
model.fit(x_train, y_train)

# Use the trained model to make predictions on the testing data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model by comparing the predicted labels with the true labels
score = accuracy_score(y_predict, y_test)

# Print the accuracy of the model as a percentage
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model to a pickle file for later use
f = open('model.p', 'wb')  # Open a file in write-binary mode
pickle.dump({'model': model}, f)  # Serialize and save the model object to the file
f.close()  # Close the file
