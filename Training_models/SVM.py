import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load the syllabification dataset
df = pd.read_csv('../dataset/dataset.csv')

# Separate the features (X) and target (y)
X = df['word']
y = df['syllable']

dtree = DecisionTreeClassifier()

print(cross_val_score(dtree, X, y, scoring="accuracy", cv=7))
mean_score = cross_val_score(dtree, X, y, scoring="accuracy", cv=7).mean()
std_score = cross_val_score(dtree, X, y, scoring="accuracy", cv=7).std()
print(mean_score)
print(std_score)
