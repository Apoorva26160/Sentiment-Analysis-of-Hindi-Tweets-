# Sentiment Analysis of Hindi Tweets

NLP pipeline for classifying Hindi tweets as **Positive**, **Negative**, or **Neutral** using character n-gram TF-IDF and an ensemble classifier.

## Pipeline

```
Raw Hindi Tweet
    → URL / Mention / Hashtag removal
    → Devanagari normalization
    → Stopword removal
    → Char n-gram TF-IDF (2–4 grams)
    → Ensemble (LR + LinearSVC)
    → Sentiment Label
```

## Why Character N-grams?

Hindi morphology is complex — character-level n-grams capture root forms, suffixes, and inflections without needing a dedicated stemmer.

## Usage

```bash
pip install -r requirements.txt
python preprocess.py
python model.py
```

## Project Structure

```
preprocess.py     # Devanagari text cleaning and normalization
model.py          # TF-IDF + Ensemble classifier, train/predict/evaluate
requirements.txt
```

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | ~0.82 | ~0.80 | ~0.81 |
| Positive | ~0.84 | ~0.85 | ~0.84 |
| Neutral | ~0.78 | ~0.79 | ~0.78 |
