# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
#
# # Load data
# data = pd.read_csv('dataset.csv')
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data['word'], data['syllables'], test_size=0.2, random_state=42)
#
# # Vectorize input data
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)
# X_train = vectorizer.fit_transform(X_train)
# X_test = vectorizer.transform(X_test)
#
# # Train SVM classifier
# clf = SVC(kernel='linear', C=1, random_state=42)
# clf.fit(X_train, y_train)
#
# # Test classifier
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# #Accuracy: 0.0

from sklearn.svm import SVC
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

# Initialize an SVM classifier with a linear kernel
svm = SVC(kernel='linear')

# Train the classifier on the training set
svm.fit(X_train, y_train)

# Use the trained classifier to predict the syllables of the test set
y_pred = svm.predict(X_test)

# Compute the accuracy of the predictions
accuracy = metrics.accuracy_score(y_test, y_pred)

# Print the accuracy of the model
print("Accuracy:", accuracy)
# Define a new list of words to predict the syllables for
new_words = ['abadiy', 'abadiylash', 'badr']

# Transform the new words into feature vectors using the CountVectorizer object
X_new = vectorizer.transform(new_words)

# Use the trained SVM classifier to predict the syllables of the new words
y_pred_new = svm.predict(X_new)

# Print the predicted syllables of the new words
print("Predicted syllables:", y_pred_new)

# Accuracy: 0.00125
# Predicted syllables: ['a-ba-diy' 'a-ba-diy-lash' 'badr']