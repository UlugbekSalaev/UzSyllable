# model for text-related tasks is the Multinomial Naive Bayes classifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load your dataset from CSV
df = pd.read_csv("../dataset/newsyllableCV.csv")

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize words and syllables using CountVectorizer
vectorizer_words = CountVectorizer(analyzer='char', lowercase=False)
X_train_words = vectorizer_words.fit_transform(train_data['word'])
X_test_words = vectorizer_words.transform(test_data['word'])

vectorizer_syllables = CountVectorizer(analyzer='char', lowercase=False)
y_train_syllables = vectorizer_syllables.fit_transform(train_data['syllables'])
y_test_syllables = vectorizer_syllables.transform(test_data['syllables'])

# Convert to numpy array using np.asarray
X_train_words = np.asarray(X_train_words.toarray())
X_test_words = np.asarray(X_test_words.toarray())
y_train_syllables = np.asarray(y_train_syllables.toarray())
y_test_syllables = np.asarray(y_test_syllables.toarray())

# Ensure the same set of unique classes in both training and testing sets
common_classes = set(y_train_syllables.argmax(axis=1)).intersection(set(y_test_syllables.argmax(axis=1)))

# Filter the data to include only common classes
train_mask = np.isin(y_train_syllables.argmax(axis=1), list(common_classes))
test_mask = np.isin(y_test_syllables.argmax(axis=1), list(common_classes))

X_train_words = X_train_words[train_mask]
y_train_syllables = y_train_syllables[train_mask]

X_test_words = X_test_words[test_mask]
y_test_syllables = y_test_syllables[test_mask]

# Define and train the Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_words, np.asarray(y_train_syllables.argmax(axis=1)))

# Predict on the test set
y_pred = nb_model.predict(X_test_words)

# Convert predictions back to syllables
predicted_syllables = vectorizer_syllables.inverse_transform(y_pred.reshape(-1, 1))

# Evaluate the model
accuracy = accuracy_score(np.asarray(y_test_syllables.argmax(axis=1)), y_pred)
print(f"Test Accuracy: {accuracy}")

# Classification Report
print("Classification Report:")
print(classification_report(np.asarray(y_test_syllables.argmax(axis=1)), y_pred))
