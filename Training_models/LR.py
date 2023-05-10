import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Step 1: Load the dataset
data = pd.read_csv('dataset.csv')

# Step 2: Split the syllables into a list
data['syllables'] = data['syllables'].apply(lambda x: x.split('-') if isinstance(x, str) else [])

# Step 3: Create a function to extract the order of vowels and consonants in a word
def extract_order(word):
    vowels = ['a', 'e', 'i', 'o', 'u', 'ō']
    order = ''
    for letter in word:
        if letter.lower() in vowels:
            order += 'V'
        else:
            order += 'C'
    return order

# Step 4: Add a column to the dataset with the order of vowels and consonants in each word
data['order'] = data['word'].apply(extract_order)

# Step 5: Encode the order column using LabelEncoder
encoder = LabelEncoder()
data['order_encoded'] = encoder.fit_transform(data['order'])

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['order_encoded']], data['syllables'], test_size=0.2)

# Step 7: Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 8: Predict the syllables of a given word
word = input('Enter a word: ')
order = extract_order(word)
order_encoded = encoder.transform([order])[0]
syllables = model.predict([order_encoded])[0]
print('Predicted syllables:', '-'.join(syllables))
