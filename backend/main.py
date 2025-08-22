# backend/main.py
import random
import joblib
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- Load GoEmotions dataset ---
dataset = load_dataset("go_emotions", "simplified")

# --- Map GoEmotions labels to our 4 categories ---
emotion_map = {
    "joy": "happy",
    "amusement": "happy",
    "approval": "happy",
    "love": "happy",
    "optimism": "happy",
    "gratitude": "happy",
    "pride": "happy",

    "sadness": "sad",
    "disappointment": "sad",
    "remorse": "sad",
    "grief": "sad",

    "anger": "angry",
    "annoyance": "angry",
    "disapproval": "angry",

    "neutral": "neutral",
}

label_names = dataset["train"].features["labels"].feature.names

texts = []
labels = []

for split in ["train", "validation", "test"]:
    for row in dataset[split]:
        emotion_ids = row["labels"]  # list[int]
        if not emotion_ids:
            continue
        mapped_labels = [
            emotion_map[label_names[e]]
            for e in emotion_ids
            if label_names[e] in emotion_map
        ]
        if mapped_labels:
            texts.append(row["text"])
            labels.append(mapped_labels[0])  # pick first valid mapping

# --- Balance the dataset ---
from collections import Counter

counts = Counter(labels)
min_count = min(counts.values())

balanced_texts = []
balanced_labels = []
for label in set(labels):
    samples = [(t, l) for t, l in zip(texts, labels) if l == label]
    chosen = random.sample(samples, min_count)
    for t, l in chosen:
        balanced_texts.append(t)
        balanced_labels.append(l)

print("Class distribution (balanced):", Counter(balanced_labels))

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    balanced_texts, balanced_labels,
    test_size=0.2, random_state=42, stratify=balanced_labels
)

# --- TF-IDF + Logistic Regression ---
clf = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2, max_df=0.9)),
    ("logreg", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)),
])

clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Save model ---
joblib.dump(clf, "sentiment_model.joblib")
print("✅ Saved model -> sentiment_model.joblib")
