# Dataset load code

import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download nltk data

nltk.download('punkt')
nltk.download('stopwords')


# Load dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# combine datasets
data = pd.concat([fake, true])
print("Dataset Loaded successfully")
print(data.head())

# Processing

stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(str(text).lower())
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered)

data["clean_text"] = data["text"].apply(preprocess)

print("\nClean Text Example:")
print(data["clean_text"].head())

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features = 5000)

x = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction 
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Test Custome News
news = ["Government announces new economic policy"]

clean = [preprocess(text) for text in news]

vector = vectorizer.transform(clean)

prediction = model.predict(vector)

print("\nPrediction:", prediction)