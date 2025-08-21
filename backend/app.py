# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained TF-IDF + Logistic Regression pipeline
clf = joblib.load("sentiment_model.joblib")

class TextInput(BaseModel):
    text: str

app = FastAPI(title="Sentiment & Emotion Classifier (TF-IDF)")

@app.post("/predict")
def predict(input: TextInput):
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
