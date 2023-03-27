from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

# Read in the dataset of words and syllables
data = pd.read_csv('dataset.csv')

# Separate the words and syllables into separate arrays
words = data['word']
syllables = data['syllables']

# Create a CountVectorizer object to convert the words into feature vectors
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2))

# Fit the vectorizer to the words in the dataset
vectorizer.fit(words)

# Transform the words into feature vectors
X = vectorizer.transform(words)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, syllables, test_size=0.2, random_state=42)

# Initialize an ANN classifier with one hidden layer of 50 neurons
ann = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)

# Train the classifier on the training set
ann.fit(X_train, y_train)

# Use the trained classifier to predict the syllables of the test set
y_pred = ann.predict(X_test)

# Compute the accuracy of the predictions
accuracy = metrics.accuracy_score(y_test, y_pred)

# Print the accuracy of the model
print("Accuracy:", accuracy)

# Define a new list of words to predict the syllables for
new_words = ["a'lochi", 'bahodir', 'balandlashuv']

# Transform the new words into feature vectors using the CountVectorizer object
X_new = vectorizer.transform(new_words)

# Use the trained ANN classifier to predict the syllables of the new words
y_pred_new = ann.predict(X_new)

# Print the predicted syllables of the new words
print("Predicted syllables:", y_pred_new)

# Accuracy: 0.00125
# Predicted syllables: ["a'-lo-chi" 'ba-ho-dir-lik' 'ba-land-la-shuv']