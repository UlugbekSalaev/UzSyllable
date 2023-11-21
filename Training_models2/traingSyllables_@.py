import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
filename = '../dataset/newsyllable.csv'
df = pd.read_csv(filename, header=None, names=['Word', 'Syllables'], encoding='latin1')  # Specify encoding

# Handle missing values
df = df.dropna()

# Tokenize words and syllables
tokenizer_words = Tokenizer(char_level=True)
tokenizer_words.fit_on_texts(df['Word'])
word_sequences = tokenizer_words.texts_to_sequences(df['Word'])
max_word_length = max(len(word) for word in word_sequences)
word_sequences_padded = pad_sequences(word_sequences, maxlen=max_word_length, padding='post')

tokenizer_syllables = Tokenizer(char_level=True)
tokenizer_syllables.fit_on_texts(df['Syllables'])
syllable_sequences = tokenizer_syllables.texts_to_sequences(df['Syllables'])
max_syllable_length = max(len(syllable) for syllable in syllable_sequences)
syllable_sequences_padded = pad_sequences(syllable_sequences, maxlen=max_syllable_length, padding='post')

# Concatenate word and syllable sequences
X = np.concatenate([word_sequences_padded, syllable_sequences_padded], axis=1)

# Create labels (target) based on syllable count
y = df['Syllables'].apply(lambda x: len(x.split('-')))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# Open a strategy scope
with strategy.scope():
    # Build a simple model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer_words.word_index)+1, output_dim=32, input_length=X.shape[1]))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation='linear'))  # Linear activation for regression

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('syllable_model.h5')
