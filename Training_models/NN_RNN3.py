import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Data Preprocessing
data = pd.read_csv('../dataset/dataset_cyrillic.csv')
# preprocess data as needed (e.g., remove duplicates, convert all words to lowercase)
# data['word'] = data['word'].apply(lambda x: re.sub('[^a-z]', '', x.lower()))
data['num_vowels'] = data['word'].apply(lambda x: sum([1 for char in x if char in ['а', 'и', 'е', 'о', 'у', 'ё', 'ю', 'я', 'э']]))
train_data, test_data = train_test_split(data, test_size=0.2)

# Step 2: Feature Extraction
X_train = np.array(train_data['num_vowels'].values.tolist())
y_train = np.array(train_data['count_syllables'].values.tolist())
X_test = np.array(test_data['num_vowels'].values.tolist())
y_test = np.array(test_data['count_syllables'].values.tolist())

# Step 3: Training the Model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1,)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[es])

# Step 4: Evaluating the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred.round())
precision = precision_score(y_test, y_pred.round(), average='weighted')
recall = recall_score(y_test, y_pred.round(), average='weighted')
f1 = f1_score(y_test, y_pred.round(), average='weighted')
print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Step 5: Using the Model
while True:
    new_word = input()
    new_word = re.sub('[^a-z]', '', new_word.lower())
    new_word_vowels = sum([1 for char in new_word if char in ['а', 'и', 'е', 'о', 'у', 'ё', 'ю', 'я', 'э']])
    predicted_syllables = model.predict(np.array([new_word_vowels]).reshape(1,1))[0][0]
    print(f'The word "{new_word}" has an estimated {round(predicted_syllables)} syllables.')
