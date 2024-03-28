import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Load the data
df = pd.read_csv('C:\\Users\\Admin\\Desktop\\prog\\python\\Smarket.csv')

# Prepare the feature matrix and target vector
X = df[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
y = (df['Direction'] == 'Up').astype(int)  # Convert 'Direction' to binary format

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Gaussian Naive Bayes model to the data
model = GaussianNB()
model.fit(X_train, y_train)

# Print the unique classes, prior probabilities, mean and variance
print('Classes:', model.classes_)
print('Prior probabilities:', model.class_prior_)
print('Mean of each feature per class:', model.theta_)
print('Variance of each feature per class:', model.var_)

# Predict the probabilities of each class for the first five samples in the test set
print('Predicted probabilities for the first five samples:', model.predict_proba(X_test[:5]))

# Predict labels for the test set
y_pred = model.predict(X_test)

# Generate a confusion matrix comparing the predicted labels with the actual labels
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:', cm)
