#  1D Convolutional Neural Network (CNN)
#  error barib turubdi
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Ensure that the padding of sequences is consistent
max_seq_length = max(X_train_words_padded.shape[1], X_test_words_padded.shape[1])

X_train_words_padded = pad_sequences(X_train_words, maxlen=max_seq_length)
X_test_words_padded = pad_sequences(X_test_words, maxlen=max_seq_length)

# Ensure that the padding of syllables is consistent
y_train_syllables_padded = pad_sequences(y_train_syllables, maxlen=max_seq_length)
y_test_syllables_padded = pad_sequences(y_test_syllables, maxlen=max_seq_length)

# Ensure that the padding of sequences is consistent with the number of units in the output layer
num_syllables = len(tokenizer_syllables.word_index) + 1

# Define the model with 1D CNN layer
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer_words.word_index) + 1, output_dim=100, input_length=X_train_words_padded.shape[1]))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(num_syllables, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_words_padded, y_train_syllables_padded, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
accuracy = model.evaluate(X_test_words_padded, y_test_syllables_padded)[1]
print(f"Test Accuracy: {accuracy}")
