# backend/main.py
import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import random

# --- Load NLTK movie reviews (map pos->happy, neg->sad) ---
texts = [' '.join(movie_reviews.words(fid)) for fid in movie_reviews.fileids()]
labels = [
    "happy" if movie_reviews.categories(fid)[0] == "pos" else "sad"
    for fid in movie_reviews.fileids()
]

# --- Base extra samples ---
extra_data = {
    "happy": [
        "I am so happy and excited!",
        "This is the best day of my life!",
        "I loved the experience, it was wonderful.",
        "Everything feels amazing right now!",
        "That movie made me smile and laugh."
    ],
    "sad": [
        "I feel really sad and depressed.",
        "I am heartbroken and lonely.",
        "Life feels empty and dark.",
        "I cried so much, I feel broken.",
        "Nothing makes me happy anymore."
    ],
    "neutral": [
        "I am just okay, nothing special.",
        "It was an average experience.",
        "Nothing much to say, just normal.",
        "The situation is fine, nothing unusual.",
        "I don’t feel strongly about this."
    ],
    "angry": [
        "I am extremely angry right now!",
        "This makes me furious and upset.",
        "I can’t control my rage anymore.",
        "That situation made me so mad!",
        "I feel so frustrated and irritated."
    ]
}

# --- Simple augmentation: shuffle, add variations ---
augmented_texts = []
augmented_labels = []

def augment_texts(label, samples, n=200):  
    for _ in range(n):  
        base = random.choice(samples)
        variation = random.choice([
            "", "!", "!!", ".", " so much", " really", " absolutely"
        ])
        augmented_texts.append(base + variation)
        augmented_labels.append(label)

# Generate ~200 samples per emotion class
for emotion, samples in extra_data.items():
    augment_texts(emotion, samples, n=200)

# --- Combine with mapped NLTK dataset ---
texts += augmented_texts
labels += augmented_labels

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- Pipeline: TF-IDF + Logistic Regression ---
clf = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2, max_df=0.9)),
    ("logreg", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)),
])

# --- Train model ---
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

# --- Save model ---
joblib.dump(clf, "sentiment_model.joblib")
print("✅ Saved model -> sentiment_model.joblib")
