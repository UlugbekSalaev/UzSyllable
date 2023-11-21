# Random Forest classifier,
#     accuracy                           0.98     12725
#    macro avg       0.66      0.69      0.67     12725
# weighted avg       0.97      0.98      0.97     12725

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('../dataset/newsyllableCV.csv')  # Replace 'your_second_dataset.csv' with the actual file path

# Encode C and V characters using LabelEncoder
le_syllable = LabelEncoder()

df['word'] = df['word'].apply(lambda x: ' '.join(list(x)))
df['syllables'] = le_syllable.fit_transform(df['syllables'])

# Convert sequences of C and V characters into a feature matrix
X = df['word'].apply(lambda x: [1 if c == 'C' else 0 for c in x.split()])
X = pd.DataFrame(X.tolist())  # Convert list of lists to DataFrame
y = df['syllables'].tolist()

# Handle missing values (NaN) with imputation
imputer = SimpleImputer(strategy='mean')  # You can adjust the strategy based on your data
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model (Random Forest as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict syllables for the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))
