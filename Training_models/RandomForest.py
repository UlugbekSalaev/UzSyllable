import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('dataset.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['word'], data['syllables'], test_size=0.2, random_state=42)

# Extract features from the data
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a Random Forest classifier on the data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict the syllables for the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

new_words = ['mashrutali', 'abadiylash', 'bajarilmoq']

# Transform the new words into feature vectors using the CountVectorizer object
X_new = vectorizer.transform(new_words)

# Use the trained ANN classifier to predict the syllables of the new words
y_pred_new = clf.predict(X_new)

# Print the predicted syllables of the new words
print("Predicted syllables:", y_pred_new)

# Accuracy: 0.0
# Predicted syllables: ['mash-ru-ta-li' 'a-ba-diy-lash' 'ba-ja-ril-moq']