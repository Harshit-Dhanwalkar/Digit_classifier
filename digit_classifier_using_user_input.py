import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load MNIST dataset
mnist = fetch_openml('mnist_784')

x, y = mnist['data'], mnist['target']

# Convert y to integers
y = y.astype(np.int8)

# Split data into train and test sets
x_train, x_test = x[:60000], x[60000:] 
y_train, y_test = y[:60000], y[60000:]

# Shuffle training data
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train.iloc[shuffle_index], y_train.iloc[shuffle_index]

# Train logistic regression model
clf = LogisticRegression(tol=0.1)
clf.fit(x_train, y_train)  # Train the model to predict the actual digit, not just 1

# Calculate accuracy on the training set
train_accuracy = clf.score(x_train, y_train)
print("Accuracy on training set:", train_accuracy)

# Calculate cross-validation scores
cv_scores = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
print("Cross-validation scores:", cv_scores)

# Calculate average accuracy
average_accuracy = np.mean(cv_scores)
print("Average accuracy:", average_accuracy)

# Function to display image and make prediction
def display_and_predict(image_index, x_data, y_data, model):
    # Fetch the image
    some_digit = x_data.iloc[image_index]
    some_digit_image = some_digit.values.reshape(28, 28)
    
    # Display the image
    plt.imshow(some_digit_image, cmap = plt.cm.binary, interpolation='nearest')
    plt.axis('on')
    plt.show()
    
    # Make prediction using the model
    prediction = model.predict([some_digit])
    print("Your predicted number is:", prediction[0])

# User input for image selection
user_input = int(input("Enter a number between 0 and 69999: "))

# Check if the input is within the range
if 0 <= user_input < 70000:
    # Display image and make prediction
    display_and_predict(user_input, x, y, clf)
else:
    print("Invalid input. Please enter a number between 0 and 69999.")
