import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords") # Download NLTK assets only once
nltk.download("punkt")
nltk.download("wordnet")

real_news = pd.read_csv("True.csv")  # Load data
fake_news = pd.read_csv("Fake.csv")

real_news["label"] = 1  # Label and merge
fake_news["label"] = 0
data = pd.concat([real_news, fake_news], ignore_index=True).sample(frac=1, random_state=42)

stop_words = set(stopwords.words("english"))  # Preprocessing
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

data["text"] = (data["title"] + " " + data["text"]).apply(clean_text)  # Apply preprocessing
data = data.drop(columns=["title", "subject", "date"], errors="ignore")

from sklearn.feature_extraction.text import TfidfVectorizer  # Feature extraction
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(data["text"]).toarray()
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression  # Model training
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2%}")

import pickle  # Save model and vectorizer
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("✅ Model and vectorizer saved successfully!")

sample = ["Factbox - Spongebob is planning to wage a nuclear war on dubai on february 30 1997"] # Enter a sample news to check
sample_vec = vectorizer.transform(sample).toarray()
result = model.predict(sample_vec)
print("📰 Prediction:", "Fake News" if result[0] == 0 else "Real News")

