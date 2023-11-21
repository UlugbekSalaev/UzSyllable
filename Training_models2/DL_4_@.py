# pip install tensorflow-addons
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout
from tensorflow.keras.models import Model
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
X_train_words_padded = pad_sequences(X_train_words, maxlen=25)
X_test_words_padded = pad_sequences(X_test_words, maxlen=25)

# Ensure that the padding of sequences is consistent with the number of units in the output layer
y_train_syllables_padded = pad_sequences(y_train_syllables, maxlen=25)
y_test_syllables_padded = pad_sequences(y_test_syllables, maxlen=25)

# Ensure that the padding of sequences is consistent with the number of units in the output layer
num_syllables = len(tokenizer_syllables.word_index) + 1


# Define the transformer model
def transformer_model(input_shape, output_shape, num_heads=4, ff_dim=32, dropout=0.1):
    inputs = Input(shape=(input_shape,))

    embedding_layer = Embedding(input_dim=len(tokenizer_words.word_index) + 1, output_dim=100, input_length=25)(inputs)

    transformer_block = tfa.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=ff_dim, dropout=dropout
    )(embedding_layer, embedding_layer)
    transformer_block = tfa.layers.FeedForwardNetwork(
        units=ff_dim, dropout=dropout
    )(transformer_block)
    transformer_block = Dropout(dropout)(transformer_block)

    output_layer = Dense(output_shape, activation='softmax')(transformer_block)

    model = Model(inputs=inputs, outputs=output_layer)
    return model


# Create the transformer model
model = transformer_model(input_shape=25, output_shape=num_syllables)

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
