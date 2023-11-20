# unidirectional LSTM
'''
Epoch 1/5
7752/7752 [==============================] - 795s 89ms/step - loss: 0.6062 - accuracy: 0.7492 - val_loss: 0.5888 - val_accuracy: 0.7569
Epoch 2/5
7752/7752 [==============================] - 525s 68ms/step - loss: 0.5929 - accuracy: 0.7550 - val_loss: 0.5873 - val_accuracy: 0.7572
Epoch 3/5
7752/7752 [==============================] - 454s 59ms/step - loss: 0.5887 - accuracy: 0.7571 - val_loss: 0.6025 - val_accuracy: 0.7531
Epoch 4/5
7752/7752 [==============================] - 446s 58ms/step - loss: 0.5867 - accuracy: 0.7579 - val_loss: 0.5865 - val_accuracy: 0.7583
Epoch 5/5
7752/7752 [==============================] - 477s 62ms/step - loss: 0.5880 - accuracy: 0.7571 - val_loss: 0.5902 - val_accuracy: 0.7537
2423/2423 [==============================] - 53s 14ms/step - loss: 0.8127 - accuracy: 0.6712
Test Accuracy: 0.6711512804031372
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your dataset from CSV
df = pd.read_csv("../dataset/newsyllableCV.csv", encoding="utf8")

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
model.add(LSTM(100, return_sequences=True))
model.add(Dense(num_syllables, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_words_padded, y_train_syllables_padded, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
accuracy = model.evaluate(X_test_words_padded, y_test_syllables_padded)[1]
print(f"Test Accuracy: {accuracy}")
