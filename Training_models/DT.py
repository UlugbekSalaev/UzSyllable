from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
data = pd.read_csv('dataset.csv')

# Split dataset into input and output
X = data.iloc[:, 0].apply(lambda x: [int(c) for c in x])
y = data['syllables'].apply(lambda x: len(x.split('-')))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set and print results
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Predicted: {y_pred}")
print(f"Accuracy: {accuracy}")
