import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the vowels and consonants
vowels = ['a', 'i', 'e', 'o', 'u', 'ō']
consonants = ['q', 'r', 't', 'y', 'p', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'v', 'b', 'n', 'm', 'ḡ', 'ş', 'ç', '’', '-']

# Load the dataset
df = pd.read_csv('data.csv')

# Preprocess the dataset
df['syllables'] = df['syllables'].str.split('-')
df = df.explode('syllables') # Flatten the list of syllables
df['word_len'] = df['word'].apply(len)
df['vowels'] = df['word'].apply(lambda x: ''.join(filter(lambda c: c in vowels, x)))
df['consonants'] = df['word'].apply(lambda x: ''.join(filter(lambda c: c in consonants, x)))

# One-hot encode the vowel and consonant orders
vowel_ohe = OneHotEncoder(categories=[range(len(vowels))])
consonant_ohe = OneHotEncoder(categories=[range(len(consonants))])

df_vowels = pd.DataFrame(vowel_ohe.fit_transform(df['vowels'].apply(lambda x: [vowels.index(c) for c in x]).tolist()).toarray(), columns=[f'vowel_{i}' for i in range(len(vowels))], index=df.index)
df_consonants = pd.DataFrame(consonant_ohe.fit_transform(df['consonants'].apply(lambda x: [consonants.index(c) for c in x]).tolist()).toarray(), columns=[f'consonant_{i}' for i in range(len(consonants))], index=df.index)

df = pd.concat([df, df_vowels, df_consonants], axis=1)
df = df.drop(['vowels', 'consonants'], axis=1)

# Encode the target variable
le = LabelEncoder()
df['syllables'] = le.fit_transform(df['syllables'])

# Split the dataset into training and testing sets
X = df.drop(['word', 'syllables'], axis=1)
y = df['syllables']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the performance of the model on the testing set
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

new_word = 'olam'
vowels_in_word = ''.join(filter(lambda c: c in vowels, new_word))
consonants_in_word = ''.join(filter(lambda c: c in consonants, new_word))
vowel_order = ''.join([str(vowels.index(c)) for c in vowels_in_word])
consonant_order = ''.join([str(consonants.index(c)) for c in consonants_in_word])
word_len = len(new_word)
features = [[vowel_order, consonant_order, word_len]]
syllable_idx = rf.predict(features)[0]
syllables = le.inverse_transform([syllable_idx])[0]
print(f"Syllables of '{new_word}': {syllables.split('-')}")
