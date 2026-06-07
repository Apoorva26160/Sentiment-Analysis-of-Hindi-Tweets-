"""
Hindi Tweet Sentiment Classifier
Trains and evaluates a sentiment model on Hindi tweets.
Labels: Positive (1), Negative (0), Neutral (2)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

from preprocess import preprocess_dataframe


LABEL_MAP = {0: "Negative", 1: "Positive", 2: "Neutral"}


def build_pipeline(classifier="ensemble"):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=50000,
        sublinear_tf=True,
    )

    if classifier == "lr":
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    elif classifier == "svm":
        clf = LinearSVC(max_iter=2000, C=1.0, class_weight="balanced")
    elif classifier == "ensemble":
        clf = VotingClassifier(estimators=[
            ("lr", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
            ("svm", LinearSVC(max_iter=2000, C=1.0, class_weight="balanced")),
        ], voting="hard")
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train(df, text_col="clean_text", label_col="label", test_size=0.2):
    df = preprocess_dataframe(df, text_col="text") if "clean_text" not in df.columns else df
    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    pipeline = build_pipeline("ensemble")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)]))

    return pipeline, X_test, y_test, y_pred


def predict(pipeline, texts):
    if isinstance(texts, str):
        texts = [texts]
    texts_clean = [preprocess_dataframe(pd.DataFrame({"text": [t]}))["clean_text"].iloc[0] for t in texts]
    preds = pipeline.predict(texts_clean)
    return [LABEL_MAP[p] for p in preds]


def save_model(pipeline, path="sentiment_model.joblib"):
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


def load_model(path="sentiment_model.joblib"):
    return joblib.load(path)


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    sample_data = pd.DataFrame({
        "text": [
            "यह फिल्म बहुत अच्छी थी मुझे बहुत पसंद आई",
            "यह बेकार है मुझे बिल्कुल पसंद नहीं",
            "ठीक है कुछ खास नहीं",
        ] * (n // 3 + 1),
        "label": ([1, 0, 2] * (n // 3 + 1))[:n],
    })

    pipeline, X_test, y_test, y_pred = train(sample_data)

    test_tweets = ["यह फिल्म शानदार थी!", "बहुत बुरा अनुभव था।"]
    print("\n=== Predictions ===")
    for tweet, sentiment in zip(test_tweets, predict(pipeline, test_tweets)):
        print(f"  '{tweet}' → {sentiment}")
