'''
Bidirectional LSTM (BiLSTM) followed by a TimeDistributed Dense layer
This model architecture introduces a Bidirectional LSTM layer, which processes the input sequence from both forward and backward directions. The TimeDistributed Dense layer is used to apply the Dense layer to each time step independently.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Load your dataset from CSV
df = pd.read_csv("../dataset/dataset_latinCV_do.csv")

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize words and syllables
tokenizer_words = Tokenizer(char_level=True)
tokenizer_words.fit_on_texts(train_data['word'])

tokenizer_syllables = Tokenizer(char_level=True)
tokenizer_syllables.fit_on_texts(train_data['syllables'])

# Convert words and syllables to sequences
X_train_words = tokenizer_words.texts_to_sequences(train_data['word'])
X_test_words = tokenizer_words.texts_to_sequences(test_data['word'])

y_train_syllables = tokenizer_syllables.texts_to_sequences(train_data['syllables'])
y_test_syllables = tokenizer_syllables.texts_to_sequences(test_data['syllables'])

# Pad sequences to the same length
X_train_words_padded = pad_sequences(X_train_words)
X_test_words_padded = pad_sequences(X_test_words)

# Ensure that the padding of sequences is consistent with the number of units in the output layer
num_syllables = len(tokenizer_syllables.word_index) + 1

y_train_syllables_padded = pad_sequences(y_train_syllables, maxlen=X_train_words_padded.shape[1])
y_test_syllables_padded = pad_sequences(y_test_syllables, maxlen=X_test_words_padded.shape[1])

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer_words.word_index) + 1, output_dim=100, input_length=X_train_words_padded.shape[1]))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(TimeDistributed(Dense(num_syllables, activation='softmax')))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_words_padded, y_train_syllables_padded, epochs=5, batch_size=32, validation_split=0.2)

# # Evaluate the model on the test set
# loss, accuracy = model.evaluate(X_test_words_padded, y_test_syllables_padded)
# print(f"Model Lost, Accuracy on Test Set: {loss}, {accuracy}")

# Evaluate the model on the test set
predictions = model.predict(X_test_words_padded)
predicted_syllables_indices = predictions.argmax(axis=-1)
y_test_syllables_flat = y_test_syllables_padded.flatten()
predicted_syllables_flat = predicted_syllables_indices.flatten()

# Calculate metrics
accuracy = accuracy_score(y_test_syllables_flat, predicted_syllables_flat)
precision = precision_score(y_test_syllables_flat, predicted_syllables_flat, average='weighted')
recall = recall_score(y_test_syllables_flat, predicted_syllables_flat, average='weighted')
f1 = f1_score(y_test_syllables_flat, predicted_syllables_flat, average='weighted')

# Print the results
print(f"Model Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Classification Report
# print("Classification Report:")
# print(classification_report(y_test_syllables_flat, predicted_syllables_flat))

while False:
    # Example input word for testing
    input_word = input("Word=")

    # Tokenize the input word
    input_word_sequence = tokenizer_words.texts_to_sequences([input_word])

    # Pad the sequence to the same length as in training
    input_word_padded = pad_sequences(input_word_sequence, maxlen=X_train_words_padded.shape[1])

    # Predict syllables for the input word
    predicted_syllables_sequence = model.predict(input_word_padded)

    # Convert the predicted syllables sequence to text
    predicted_syllables = tokenizer_syllables.sequences_to_texts(predicted_syllables_sequence.argmax(axis=-1))[0]

    # Print the results
    print(f"Input Word: {input_word}")
    print(f"Predicted Syllables: {predicted_syllables}")