#  Seq2Seq model using LSTM layers
'''
Epoch 1/5
8722/8722 [==============================] - 162s 18ms/step - loss: 0.4449 - accuracy: 0.8650 - val_loss: 0.3229 - val_accuracy: 0.9060
Epoch 2/5
8722/8722 [==============================] - 221s 25ms/step - loss: 0.1730 - accuracy: 0.9468 - val_loss: 0.1664 - val_accuracy: 0.9497
Epoch 3/5
8722/8722 [==============================] - 212s 24ms/step - loss: 0.0835 - accuracy: 0.9748 - val_loss: 0.1152 - val_accuracy: 0.9648
Epoch 4/5
8722/8722 [==============================] - 245s 28ms/step - loss: 0.0523 - accuracy: 0.9846 - val_loss: 0.0876 - val_accuracy: 0.9730
Epoch 5/5
8722/8722 [==============================] - 218s 25ms/step - loss: 0.0369 - accuracy: 0.9893 - val_loss: 0.0697 - val_accuracy: 0.9779
2423/2423 [==============================] - 26s 11ms/step - loss: 0.0938 - accuracy: 0.9742
Test Loss: 0.09377637505531311
Test Accuracy: 0.974197268486023
'''
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# Sample dataset
df = pd.read_csv("../dataset/newsyllable.csv")

word_sequences = df["word"]                 #['a’lohazrat', 'a’mol', 'a’moliy', 'a’moliy-tadrijiy']
syllable_sequences = df["syllables"]        #['a’-lo-haz-rat', 'a’-mol', 'a’-mo-liy', 'a’-mo-liy-tad-ri-jiy']

# Create a vocabulary of unique characters
all_chars = set(''.join(word_sequences))
all_chars.update(set(''.join(syllable_sequences)))
char2idx = {char: idx for idx, char in enumerate(all_chars)}
idx2char = {idx: char for idx, char in enumerate(all_chars)}

# Convert sequences to numerical representation
def sequence_to_indices(sequence, char2idx):
    return [char2idx[char] for char in sequence]

X = [sequence_to_indices(word, char2idx) for word in word_sequences]
y = [sequence_to_indices(syllable, char2idx) for syllable in syllable_sequences]

# Pad sequences for a consistent input size
X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post')
y = tf.keras.preprocessing.sequence.pad_sequences(y, padding='post')

# Define the Seq2Seq model
embedding_dim = 32
hidden_units = 64

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(len(char2idx), embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(hidden_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(len(char2idx), embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(char2idx), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train the model
model.fit([X_train, y_train[:, :-1]], y_train[:, 1:], epochs=5, batch_size=32, validation_split=0.1)

model.save('dl5.h5')
# Evaluate the model on the test set
eval_result = model.evaluate([X_test, y_test[:, :-1]], y_test[:, 1:])
print("Test Loss:", eval_result[0])
print("Test Accuracy:", eval_result[1])

# You can use this trained model for predictions on new data
