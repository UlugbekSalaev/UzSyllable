import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# Define and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_words, np.asarray(y_train_syllables.argmax(axis=1)))

# Predict on the test set
y_pred = knn_model.predict(X_test_words)

# Convert predictions back to syllables
predicted_syllables = vectorizer_syllables.inverse_transform(y_pred.reshape(-1, 1))

# Evaluate the model
accuracy = accuracy_score(np.asarray(y_test_syllables.argmax(axis=1)), y_pred)
print(f"Test Accuracy: {accuracy}")

# Classification Report
print("Classification Report:")
print(classification_report(np.asarray(y_test_syllables.argmax(axis=1)), y_pred))
