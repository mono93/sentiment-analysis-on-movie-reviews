# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Use a single FastAPI instance and add CORS middleware to it
app = FastAPI(title="Sentiment & Emotion Classifier (TF-IDF)")

# --- CORS setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained TF-IDF + Logistic Regression pipeline
clf = joblib.load("sentiment_model.joblib")

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze(input: TextInput):
    # Predict label
    prediction = clf.predict([input.text])[0]
    
    # Get probability scores
    probas = clf.predict_proba([input.text])[0]
    labels = clf.classes_
    
    # Confidence for predicted label
    confidence = probas[labels.tolist().index(prediction)] * 100
    
    # Format all probabilities as percentages
    all_probs = {labels[i]: f"{probas[i] * 100:.2f}%" for i in range(len(labels))}
    
    return {
        "text": input.text,
        "prediction": prediction,
        "confidence": f"{confidence:.2f}%",
        "all_probabilities": all_probs
    }
