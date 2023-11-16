import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier  # Example of a simpler model

# Load the dataset
df = pd.read_csv('../dataset/dataset_latin.csv')  # Replace 'your_dataset.csv' with the actual file path
df = df.sample(frac=0.1, random_state=42)  # Use a smaller fraction for testing

# Encode words and syllables using LabelEncoder
le_word = LabelEncoder()
le_syllable = LabelEncoder()

df['Word'] = le_word.fit_transform(df['word'])
df['Syllable'] = le_syllable.fit_transform(df['syllables'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Word'], df['Syllable'], test_size=0.2, random_state=42)

# Use a simpler model (Random Forest as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train.values.reshape(-1, 1), y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test.values.reshape(-1, 1))

# Transform predicted and true labels back to the original form
y_pred = le_syllable.inverse_transform(y_pred)
y_test = le_syllable.inverse_transform(y_test)

# Evaluate the performance (you can use your own evaluation metric)
accuracy = sum(y_pred == y_test) / len(y_test)
print(f'Accuracy: {accuracy}')