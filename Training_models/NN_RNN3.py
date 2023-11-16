import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Data Preprocessing
data = pd.read_csv('../dataset/dataset_cyrillic.csv')
# preprocess data as needed (e.g., remove duplicates, convert all words to lowercase)
# data['word'] = data['word'].apply(lambda x: re.sub('[^a-z]', '', x.lower()))
data['num_vowels'] = data['word'].apply(lambda x: sum([1 for char in x if char in ['а', 'и', 'е', 'о', 'у', 'ё', 'ю', 'я', 'э']]))

X = np.array(data['num_vowels'].values.tolist())
y = np.array(data['count_syllables'].values.tolist())

# Step 2: Model Training and Evaluation using Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(1,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[es])

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred.round())
    precision = precision_score(y_test, y_pred.round(), average='weighted')
    recall = recall_score(y_test, y_pred.round(), average='weighted')
    f1 = f1_score(y_test, y_pred.round(), average='weighted')

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Compute Mean and Standard Deviation of evaluation metrics
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)
mean_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f'Mean Accuracy: {mean_accuracy:.3f} (+/- {std_accuracy:.3f})')
print(f'Mean Precision: {mean_precision:.3f} (+/- {std_precision:.3f})')
print(f'Mean Recall: {mean_recall:.3f} (+/- {std_recall:.3f})')
print(f'Mean F1 Score: {mean_f1:.3f} (+/- {std_f1:.3f})')
