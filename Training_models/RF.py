#Random Forest algorithm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the syllabification dataset
df = pd.read_csv('../dataset/dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['word'], df['syllables'], test_size=0.2, random_state=42)

# Define a function to extract features from words
def extract_features(word):
    features = []
    features.append(len(word))
    features.append(len([char for char in word if char in ['a', 'e', 'i', 'o', 'u', 'o‘']]))
    features.append(len([char for char in word if char.isupper()]))
    features.append(len([char for char in word if char.isdigit()]))
    return features

# Extract features from the training set

X_train = [extract_features(word) for word in X_train]

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Extract features from the testing set
X_test = [extract_features(word) for word in X_test]

# Evaluate the performance of the classifier on the testing set
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
