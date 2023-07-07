# yangi lotin alifbosi uchun syllable count
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Step 1: Data Preprocessing
data = pd.read_csv('../dataset/dataset_cyrillic.csv', encoding="utf8")
# preprocess data as needed (e.g., remove duplicates, convert all words to lowercase)

#data['word'] = data['word'].apply(lambda x: re.sub('[^a-z]', '', x.lower()))
data['num_vowels'] = data['word'].apply(lambda x: sum([1 for char in x if char in ['а', 'и', 'е', 'о', 'у', 'ё', 'ю', 'я', 'э']]))
train_data, test_data = train_test_split(data, test_size=0.2)

# Step 2: Feature Extraction
X_train = train_data['num_vowels'].values.reshape(-1, 1)
y_train = train_data['count_syllables']
X_test = test_data['num_vowels'].values.reshape(-1, 1)
y_test = test_data['count_syllables']

# Step 3: Training the Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 4: Evaluating the Model
y_pred = rf_model.predict(X_test)
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
    predicted_syllables = rf_model.predict([[new_word_vowels]])[0]
    print(f'The word "{new_word}" has an estimated {predicted_syllables.round()} syllables.')