# src/infer.py
"""
Script for making predictions using the trained model.
"""
import os
import pickle
from preprocess import clean_text  # must match training-time cleaning

MODELS_DIR = "models"
VECT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")

def load_model():
    with open(VECT_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

def predict_sentiment(text, vectorizer, model, return_proba=False):
    """
    Predict sentiment for given text.
    Returns the model's label directly (string or int) to avoid mapping errors.
    """
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    # (Optional) debug – if X is all zeros often, your preprocessing doesn't match training
    # print("nnz:", X.nnz)

    pred = model.predict(X)[0]

    if return_proba and hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", None)
        if classes is not None:
            # return a dict of {class_label: probability}
            return str(pred), {str(c): float(p) for c, p in zip(classes, probs)}
        else:
            return str(pred), None

    return str(pred)  # ensure JSON-serializable

def main():
    vectorizer, model = load_model()
    samples = [
        "I absolutely love this new iPhone! Best purchase ever!",
        "This product is terrible, worst experience ever.",
        "It's okay, nothing special about it.",
        "Why is this in my feed?"
    ]
    for s in samples:
        label, proba = predict_sentiment(s, vectorizer, model, return_proba=True)
        print(f"{s} -> {label} | {proba}")

if __name__ == "__main__":
    main()
