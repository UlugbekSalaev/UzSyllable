import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the test dataset
filename = '../dataset/newsyllable.csv'
df = pd.read_csv(filename, header=None, names=['Word', 'Syllables'], encoding='latin1')
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
X_test = np.concatenate([word_sequences_padded, syllable_sequences_padded], axis=1)
print(X_test)
# Load the saved model
loaded_model = load_model('trainingSyllables.h5')

# Evaluate the model on the test data
accuracy = loaded_model.evaluate(X_test, df['Syllables'].apply(lambda x: len(x.split('-'))))
print("Accuracy:", accuracy)