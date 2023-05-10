import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from a CSV file
df = pd.read_csv('../dataset/binary.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['x'], df['y'], test_size=0.2, random_state=42)

# Define a function to convert a string of * and # into a feature vector
def str_to_vec(s):
    return [1 if c == '*' else 0 if c == "#" else "-" for c in s]

# Convert the training and testing set strings to feature vectors
X_train = [str_to_vec(s) for s in X_train]
X_test = [str_to_vec(s) for s in X_test]

# Define a function to convert a string of *, #, and - into a label vector
def str_to_label(s):
    return [c for c in s]

# Convert the training and testing set strings to label vectors
y_train = [str_to_label(s) for s in y_train]
y_test = [str_to_label(s) for s in y_test]

y_train = [str_to_vec(s) for s in y_train]
print((y_train))

# Define the decision tree classifier and train it on the training set
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Use the trained model to make predictions on the testing set
y_pred = clf.predict(X_test)

# Compute the accuracy of the model
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
