import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

# Step 1: Load the dataset
data = pd.read_csv('dataset.csv')

# Step 2: Split the syllables into a list
data['syllables'] = data['syllables'].apply(lambda x: x.split('-'))

# Step 3: Create a dictionary of vowels
vowels = {'a', 'i', 'e', 'o', 'u', 'ō'}

# Step 4: Define a function to count the number of vowels in a word
def count_vowels(word):
    return sum([1 for letter in word if letter.lower() in vowels])

# Step 5: Add a column to the dataset with the number of vowels in each word
data['vowel_count'] = data['word'].apply(count_vowels)

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['vowel_count']], data['syllables'], test_size=0.2)

# Step 7: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predict the syllables of a given word
word = input('Enter a word: ')
vowel_count = count_vowels(word)
syllables = model.predict([[vowel_count]])[0]
print('Predicted syllables:', '-'.join(syllables))
