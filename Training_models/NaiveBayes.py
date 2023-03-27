import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['word'], data['syllables'], test_size=0.2, random_state=42)

# Convert the text data into feature vectors using CountVectorizer
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Predict the syllables for the test set
y_pred = nb.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
new_words = ['baliqxo‘r', 'chuchmoma', 'bajarilmoq']

# Transform the new words into feature vectors using the CountVectorizer object
X_new = vectorizer.transform(new_words)

# Use the trained ANN classifier to predict the syllables of the new words
y_pred_new = nb.predict(X_new)

# Print the predicted syllables of the new words
print("Predicted syllables:", y_pred_new)
# Accuracy: 0.0
# Predicted syllables: ['ba-liq-xo‘r-lik' 'chuch-mo-ma-dosh-lar' 'ba-ja-ril-moq']

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import CountVectorizer
# import pandas as pd
#
# # Read in the dataset of words and syllables
# data = pd.read_csv('dataset.csv')
#
# # Separate the words and syllables into separate arrays
# words = data['word']
# syllables = data['syllables']
#
# # Create a CountVectorizer object to convert the words into feature vectors
# vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2))
#
# # Fit the vectorizer to the words in the dataset
# vectorizer.fit(words)
#
# # Transform the words into feature vectors
# X = vectorizer.transform(words)
#
# # Initialize a Multinomial Naive Bayes classifier
# nb = MultinomialNB()
#
# # Train the classifier on the feature vectors and syllable labels
# nb.fit(X, syllables)
#
# # Use the trained classifier to predict the syllables of a new word
# new_word = 'abadiy'
# new_word_vector = vectorizer.transform([new_word])
# predicted_syllables = nb.predict(new_word_vector)
#
# # Print the predicted syllables of the new word
# print(predicted_syllables) #['a-ba-diy-bo-qiy']
'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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

# Initialize a Multinomial Naive Bayes classifier
nb = MultinomialNB()

# Train the classifier on the training set
nb.fit(X_train, y_train)

# Use the trained classifier to predict the syllables of the test set
y_pred = nb.predict(X_test)

# Compute the accuracy of the predictions
accuracy = metrics.accuracy_score(y_test, y_pred)

# Print the accuracy of the model
print("Accuracy:", accuracy)
new_words = ['mashrutali', 'abadiylash', 'bajarilmoq']

# Transform the new words into feature vectors using the CountVectorizer object
X_new = vectorizer.transform(new_words)

# Use the trained ANN classifier to predict the syllables of the new words
y_pred_new = nb.predict(X_new)

# Print the predicted syllables of the new words
print("Predicted syllables:", y_pred_new)

# Accuracy: 0.0
# Predicted syllables: ['mash-ru-ta-li' 'a-ba-diy-lash-ti-rish' 'ba-ja-ril-moq']
'''