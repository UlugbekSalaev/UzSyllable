import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# Read the dataset file
df = pd.read_csv('dataset.csv')

# Extract the "word" and "syllable" columns
words = df['word'].tolist()
syllables = df['syllable'].tolist()

# Define a function to encode the dataset
def encode_dataset(words, syllables):
    # Create dictionaries for character to index and index to character mapping
    chars = sorted(list(set(''.join(words))))
    char_to_index = dict((c, i) for i, c in enumerate(chars))
    index_to_char = dict((i, c) for i, c in enumerate(chars))

    # Create input and output arrays
    X = np.zeros((len(words), len(max(words, key=len)), len(chars)), dtype=np.bool)
    y = np.zeros((len(words), len(max(syllables, key=len)), len(chars)), dtype=np.bool)
    for i, word in enumerate(words):
        for j, char in enumerate(word):
            X[i, j, char_to_index[char]] = 1
        for j, char in enumerate(syllables[i]):
            y[i, j, char_to_index[char]] = 1

    return X, y, char_to_index, index_to_char

# Encode the dataset
X, y, char_to_index, index_to_char = encode_dataset(words, syllables)

# Define the RNN model
model = Sequential()
model.add(SimpleRNN(64, input_shape=(len(max(words, key=len)), len(char_to_index))))
model.add(Dense(len(char_to_index), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000, batch_size=len(words))

# Test the model
test_word = 'python'
test_X = np.zeros((1, len(test_word), len(char_to_index)), dtype=np.bool)
for i, char in enumerate(test_word):
    test_X[0, i, char_to_index[char]] = 1
prediction = model.predict(test_X)[0]
predicted_syllables = []
for i, char in enumerate(test_word):
    if prediction[i, char_to_index['-']] > 0.5:
        predicted_syllables.append('-')
    predicted_syllables.append(char)
print(test_word + ': ' + ''.join(predicted_syllables))
