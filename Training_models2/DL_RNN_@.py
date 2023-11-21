# recurrent neural networks (RNNs) or long short-term memory networks (LSTMs) are commonly used due to their ability to capture sequential patterns. Here's an example of a simple LSTM-based neural network for syllabification using Keras
# This code uses an Embedding layer for word representation and an LSTM layer for sequence modeling. Please adjust hyperparameters, such as the number of LSTM units and embedding dimensions, based on your specific requirements and experiment as needed.
'''
Epoch 1/5
7752/7752 [==============================] - 8977s 1s/step - loss: 0.6055 - accuracy: 0.7497 - val_loss: 0.5891 - val_accuracy: 0.7567
Epoch 2/5
7752/7752 [==============================] - 206s 27ms/step - loss: 0.5975 - accuracy: 0.7523 - val_loss: 0.6453 - val_accuracy: 0.7332
Epoch 3/5
7752/7752 [==============================] - 351s 45ms/step - loss: 0.5887 - accuracy: 0.7564 - val_loss: 0.5863 - val_accuracy: 0.7573
Epoch 4/5
7752/7752 [==============================] - 266s 34ms/step - loss: 0.5881 - accuracy: 0.7572 - val_loss: 0.5855 - val_accuracy: 0.7583
Epoch 5/5
7752/7752 [==============================] - 244s 32ms/step - loss: 0.5859 - accuracy: 0.7584 - val_loss: 0.5853 - val_accuracy: 0.7587
Traceback (most recent call last):
  File "C:\Users\E-MaxPCShop\PycharmProjects\UzSyllable\Training_models2\FNN.py", line 55, in <module>
    predictions = model.predict(X_test_words_padded)
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
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
X_train_words_padded = pad_sequences(X_train_words)
X_test_words_padded = pad_sequences(X_test_words)

# Ensure that the padding of sequences is consistent with the number of units in the output layer
num_syllables = len(tokenizer_syllables.word_index) + 1

y_train_syllables_padded = pad_sequences(y_train_syllables, maxlen=X_train_words_padded.shape[1])
y_test_syllables_padded = pad_sequences(y_test_syllables, maxlen=X_test_words_padded.shape[1])

# Define the neural network model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer_words.word_index) + 1, output_dim=100, input_length=X_train_words_padded.shape[1]))
model.add(LSTM(100, return_sequences=True))
model.add(Dense(num_syllables, activation='softmax'))

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
