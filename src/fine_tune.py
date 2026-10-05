"""
Stretch Goal: Fine-Tuning & Model Comparison.

Trains a fast local sentiment and complaint theme classifier (Logistic Regression on embeddings / TF-IDF)
and provides a structured trade-off comparison matrix between RAG and Fine-Tuning.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class FineTunedReviewClassifier:
    def __init__(self):
        """Initializes a lightweight TF-IDF + LogisticRegression sentiment & complaint model."""
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=500, C=1.0))
        ])
        self.is_trained = False

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains the classifier on review text to predict Sentiment (Positive, Neutral, Negative)."""
        print("[Stretch Goal] Training fine-tuned sentiment classifier...")
        
        texts = df["CombinedReview"].tolist()
        labels = df["Sentiment"].tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        
        t0 = time.time()
        self.pipeline.fit(X_train, y_train)
        training_time = time.time() - t0
        
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.is_trained = True
        print(f"[Stretch Goal Complete] Trained in {training_time:.2f}s | Accuracy: {acc*100:.2f}%")
        
        return {
            "training_time_sec": round(training_time, 4),
            "accuracy": round(acc, 4),
            "report": report
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """Predicts sentiment category and extracts key complaint terms."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")
            
        t0 = time.time()
        pred_label = self.pipeline.predict([text])[0]
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        
        prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
        inference_time = (time.time() - t0) * 1000 # in ms
        
        # Simple complaint extractor
        complaint_keywords = ["packaging", "crushed", "burnt", "stale", "expired", "bitter", "rancid", "broken", "seal", "overpriced"]
        found_complaints = [kw for kw in complaint_keywords if kw in text.lower()]
        
        return {
            "predicted_sentiment": pred_label,
            "probabilities": prob_dict,
            "detected_complaint_themes": found_complaints if found_complaints else ["None detected"],
            "inference_time_ms": round(inference_time, 2)
        }


def get_rag_vs_finetuning_comparison() -> Dict[str, Any]:
    """
    Returns a comprehensive architectural comparison matrix between RAG and Fine-Tuning.
    """
    return {
        "comparison_matrix": [
            {
                "dimension": "Primary Use Case",
                "rag": "Open-ended QA, document search, synthesis with citations",
                "fine_tuning": "Specific classification, style transfer, domain vocabulary adaptation"
            },
            {
                "dimension": "Knowledge Update Frequency",
                "rag": "Instantaneous (Add new vector to FAISS/Chroma without re-training)",
                "fine_tuning": "Slow & Expensive (Requires dataset re-labeling and re-training run)"
            },
            {
                "dimension": "Hallucination Control",
                "rag": "High Groundedness (Constrained to retrieved context with citations)",
                "fine_tuning": "Risk of Parametric Hallucination (Generates plausible-sounding false data)"
            },
            {
                "dimension": "Inference Latency",
                "rag": "Medium to High (Vector search ~10ms + LLM Generation ~500ms - 2000ms)",
                "fine_tuning": "Ultra Low (Small classifier ~5ms on CPU)"
            },
            {
                "dimension": "Source Transparency",
                "rag": "100% Traceable (Provides exact review source IDs, product IDs, ratings)",
                "fine_tuning": "Black Box (Weights store opaque parametric knowledge)"
            },
            {
                "dimension": "Cost Structure",
                "rag": "Low training cost; API token costs per query",
                "fine_tuning": "Higher upfront GPU compute cost; ultra cheap inference"
            }
        ],
        "recommendation_framework": [
            "Use RAG when you need up-to-the-minute customer review search, exact quote evidence, or answers to unpredictable user queries.",
            "Use Fine-Tuning when you need fast, cheap, high-volume automated tagging of sentiment, intent, or routing tickets to support teams.",
            "Use a Hybrid Approach: Use a Fine-Tuned classifier to tag incoming reviews instantly, then index them into RAG for interactive AI Q&A!"
        ]
    }


if __name__ == "__main__":
    from data_loader import load_and_preprocess_reviews
    df, _ = load_and_preprocess_reviews(sample_size=500)
    clf = FineTunedReviewClassifier()
    metrics = clf.train(df)
    res = clf.predict("I bought the coffee bag but the seal was broken and it smelled burnt!")
    print("Prediction:", res)
    print("RAG vs FT Comparison:", get_rag_vs_finetuning_comparison())
