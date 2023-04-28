import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('dataset.csv')

# Extract the words and syllables from the DataFrame
words = list(df['word'])
syllables = [syllable.split('-') for syllable in df['syllables']]

# Convert words to feature vectors
X = [[len(syllable), i] for i, word in enumerate(words) for syllable in syllables[i]]
y = [i for i, word in enumerate(words) for syllable in syllables[i]]

# Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluate your model
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy) # Accuracy: 0.893719806763285

# Apply your model
# new_words = ['mashrutali', 'abadiylash', 'bajarilmoq']
# for word in new_words:
#     X_new = [[len(syllable), i] for i, syllable in enumerate(word)]
#     y_pred = clf.predict(X_new)
#     syllabification = [syllables[y_pred[0]][0]]
#     for i in range(1, len(y_pred)):
#         if y_pred[i] == y_pred[i-1]:
#             syllabification[-1] += syllables[y_pred[i]][1]
#         else:
#             syllabification.append(syllables[y_pred[i]][0])
#     print(word, '->', '-'.join(syllabification))

