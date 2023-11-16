import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
dataset_path = "../dataset/dataset_latin.csv"
df = pd.read_csv(dataset_path)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Define functions for feature extraction
def get_word_column(data):
    return data['word']

def get_syllable_column(data):
    return data['syllables']

# Machine Learning Models
svm_model = make_pipeline(
    ColumnTransformer(
        transformers=[
            ('word', CountVectorizer(analyzer='char', ngram_range=(1, 3)), 'word')
        ],
        remainder='passthrough'
    ),
    SVC()
)

rf_model = make_pipeline(
    ColumnTransformer(
        transformers=[
            ('word', CountVectorizer(analyzer='char', ngram_range=(1, 3)), 'word')
        ],
        remainder='passthrough'
    ),
    RandomForestClassifier()
)

nb_model = make_pipeline(
    ColumnTransformer(
        transformers=[
            ('word', CountVectorizer(analyzer='char', ngram_range=(1, 3)), 'word')
        ],
        remainder='passthrough'
    ),
    MultinomialNB()
)

lr_model = make_pipeline(
    ColumnTransformer(
        transformers=[
            ('word', CountVectorizer(analyzer='char', ngram_range=(1, 3)), 'word')
        ],
        remainder='passthrough'
    ),
    LogisticRegression()
)

# Deep Learning Model
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(train_data['word'])
total_words = len(tokenizer.word_index) + 1

max_sequence_length = max(train_data['word'].apply(len))
train_sequences = pad_sequences(tokenizer.texts_to_sequences(train_data['word']), maxlen=max_sequence_length, padding='post')
test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_data['word']), maxlen=max_sequence_length, padding='post')

dl_model = Sequential([
    Embedding(input_dim=total_words, output_dim=32, input_length=max_sequence_length),
    Bidirectional(LSTM(64)),
    Dense(1, activation='sigmoid')
])

# Train Machine Learning Models
ml_models = {'SVM': svm_model, 'Random Forest': rf_model, 'Naive Bayes': nb_model, 'Logistic Regression': lr_model}

for name, model in ml_models.items():
    model.fit(get_word_column(train_data), get_syllable_column(train_data))
    predictions = model.predict(get_word_column(test_data))
    accuracy = accuracy_score(get_syllable_column(test_data), predictions)
    print(f"{name} Model Accuracy: {accuracy}")
    print(f"{name} Model Classification Report:\n{classification_report(get_syllable_column(test_data), predictions)}")
    print("\n")

# Train Deep Learning Model
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(train_sequences, get_syllable_column(train_data), epochs=5, batch_size=32, validation_split=0.2)

# Evaluate Deep Learning Model
dl_predictions = (dl_model.predict(test_sequences) > 0.5).astype("int32")
dl_accuracy = accuracy_score(get_syllable_column(test_data), dl_predictions)
print(f"Deep Learning Model Accuracy: {dl_accuracy}")
print(f"Deep Learning Model Classification Report:\n{classification_report(get_syllable_column(test_data), dl_predictions)}")
