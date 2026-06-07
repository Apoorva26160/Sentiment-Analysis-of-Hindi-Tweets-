"""
Hindi Tweet Preprocessing
Handles Devanagari text cleaning, normalization, and tokenization.
"""

import re
import string
import pandas as pd


HINDI_STOPWORDS = {
    "है", "हैं", "था", "थे", "थी", "हो", "हूँ", "हुआ", "हुई", "हुए",
    "और", "या", "तो", "भी", "ही", "कि", "को", "के", "की", "का", "में",
    "से", "पर", "यह", "वह", "इस", "उस", "जो", "कर", "कह", "नहीं", "एक",
    "आप", "हम", "मैं", "वे", "यहाँ", "वहाँ", "अब", "तब", "जब", "कब",
}


def remove_urls(text):
    return re.sub(r"http\S+|www\S+", "", text)


def remove_mentions_hashtags(text, keep_hashtag_text=True):
    text = re.sub(r"@\w+", "", text)
    if keep_hashtag_text:
        text = re.sub(r"#(\w+)", r"\1", text)
    else:
        text = re.sub(r"#\w+", "", text)
    return text


def remove_english_chars(text):
    return re.sub(r"[a-zA-Z]", "", text)


def remove_punctuation(text):
    punct = string.punctuation + "।॥॰"
    return text.translate(str.maketrans("", "", punct))


def normalize_hindi(text):
    # Normalize nukta variations
    text = re.sub(r"[ऩऱऴ]", lambda m: {"ऩ": "न", "ऱ": "र", "ऴ": "ळ"}.get(m.group(), m.group()), text)
    # Collapse repeated characters
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text


def remove_stopwords(tokens):
    return [t for t in tokens if t not in HINDI_STOPWORDS and len(t) > 1]


def tokenize(text):
    return text.split()


def preprocess(text, remove_english=True):
    text = str(text).strip()
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    if remove_english:
        text = remove_english_chars(text)
    text = remove_punctuation(text)
    text = normalize_hindi(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def preprocess_dataframe(df, text_col="text"):
    df = df.copy()
    df["clean_text"] = df[text_col].apply(preprocess)
    df["token_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df = df[df["token_count"] > 0].reset_index(drop=True)
    return df


if __name__ == "__main__":
    sample = pd.DataFrame({
        "text": [
            "यह बहुत अच्छा है! #India @user http://example.com",
            "मुझे यह बिल्कुल पसंद नहीं है।",
            "Great movie! बहुत मज़ेदार था।",
        ]
    })
    result = preprocess_dataframe(sample)
    print(result[["text", "clean_text"]])
