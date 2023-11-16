import csv
import hmmlearn

# Load the necessary libraries
import pandas as pd

# Read the dataset from the CSV file
data = pd.read_csv('../dataset/dataset_latin.csv')

# Split the data into training and testing sets
train_data = data[:int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):]

# Convert the data into a format suitable for the HMM model
train_words = train_data['word'].values
train_syllables = train_data['syllables'].values

test_words = test_data['word'].values
test_syllables = test_data['syllables'].values

# Define the observation and state sequences for the HMM model
observations = []
states = []

for word, syllable in zip(train_words, train_syllables):
    observations.append([s for s in syllable])
    states.append([w for w in word])

# Create an HMM model
model = hmmlearn.HiddenMarkovModel()

# Train the HMM model
model.fit(observations, states)

# Predict the syllabification of the test data
predicted_syllables = model.predict(observations)

# Evaluate the model
accuracy = model.score(observations, states)
print("Accuracy:", accuracy)

# Print the predicted syllabification for the test data
for word, predicted_syllable in zip(test_words, predicted_syllables):
    print(f"Word: {word}, Predicted Syllable: {predicted_syllable}")
