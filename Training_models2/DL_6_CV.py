#  This model uses an embedding layer followed by a bidirectional LSTM layer and a time-distributed dense layer.
#  Bidirectional LSTM
'''
Epoch 1/5
7752/7752 [==============================] - 207s 26ms/step - loss: 0.1029 - accuracy: 0.9592 - val_loss: 0.0311 - val_accuracy: 0.9925
Epoch 2/5
7752/7752 [==============================] - 197s 25ms/step - loss: 0.0289 - accuracy: 0.9932 - val_loss: 0.0275 - val_accuracy: 0.9936
Epoch 3/5
7752/7752 [==============================] - 183s 24ms/step - loss: 0.0269 - accuracy: 0.9937 - val_loss: 0.0262 - val_accuracy: 0.9938
Epoch 4/5
7752/7752 [==============================] - 178s 23ms/step - loss: 0.0262 - accuracy: 0.9938 - val_loss: 0.0257 - val_accuracy: 0.9939
Epoch 5/5
7752/7752 [==============================] - 185s 24ms/step - loss: 0.0258 - accuracy: 0.9939 - val_loss: 0.0256 - val_accuracy: 0.9940
2423/2423 [==============================] - 13s 5ms/step
Test Accuracy: 0.9941054749987099
weighted avg       0.99      0.99      0.99   1937900
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset from CSV
df = pd.read_csv("../dataset/newsyllableCV.csv")

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
X_train_words_padded = pad_sequences(X_train_words, maxlen=25)  # Set maxlen to the desired sequence length
X_test_words_padded = pad_sequences(X_test_words, maxlen=25)

# Ensure that the padding of sequences is consistent with the number of units in the output layer
y_train_syllables_padded = pad_sequences(y_train_syllables, maxlen=25)
y_test_syllables_padded = pad_sequences(y_test_syllables, maxlen=25)

# Ensure that the padding of sequences is consistent with the number of units in the output layer
num_syllables = len(tokenizer_syllables.word_index) + 1

y_train_syllables_padded = pad_sequences(y_train_syllables, maxlen=X_train_words_padded.shape[1])
y_test_syllables_padded = pad_sequences(y_test_syllables, maxlen=X_test_words_padded.shape[1])

# Define the neural network model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer_words.word_index) + 1, output_dim=100, input_length=25))  # Adjust input_length
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(TimeDistributed(Dense(num_syllables, activation='softmax')))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_words_padded, y_train_syllables_padded, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
predictions = model.predict(X_test_words_padded)
y_pred_encoded = predictions.argmax(axis=-1)

# Convert predictions back to syllables
y_pred_syllables = tokenizer_syllables.sequences_to_texts(y_pred_encoded)

# Evaluate the model
accuracy = accuracy_score(y_test_syllables_padded.flatten(), y_pred_encoded.flatten())
print(f"Test Accuracy: {accuracy}")

# Classification Report
print("Classification Report:")
print(classification_report(y_test_syllables_padded.flatten(), y_pred_encoded.flatten()))