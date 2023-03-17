
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the dataset into a Pandas dataframe
df = pd.read_csv('dataset.csv')
# Encode the syllables column using label encoding
encoder = LabelEncoder()
df['syllables'] = encoder.fit_transform(df['syllables'])
df['word'] = encoder.fit_transform(df['word'])
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('word', axis=1), df['syllables'], test_size=0.2)
# Initialize a Decision Tree model
model = DecisionTreeClassifier()
# Train the model on the training data
model.fit(X_train, y_train)
# Use the model to make predictions on the testing data
y_pred = model.predict(X_test)
# Convert the predicted numeric values back to their original syllables using inverse_transform
y_pred_syllables = encoder.inverse_transform(y_pred)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
