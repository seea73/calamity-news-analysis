import os
import re
import pandas as pd
import spacy

# Load English model only once (much faster)
nlp = spacy.load("en_core_web_sm", disable=["ner"])  # NER disabled → faster


# ---------------------------------------
# 1. Basic cleaning function
# ---------------------------------------
def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove digits
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------
# 2. Lemmatization + stopword removal
# ---------------------------------------
def lemmatize_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop                                     # remove stopwords
        and len(token.lemma_) > 2                                # remove single chars
        and token.is_alpha                                       # keep only alphabets
    ]
    return " ".join(tokens)


# ---------------------------------------
# 3. Main pipeline
# ---------------------------------------
def main():
    input_path = "data/raw/all_news_raw.csv"
    output_dir = "data/processed"
    output_path = f"{output_dir}/news_clean.csv"

    print("\nLoading dataset:", input_path)
    df = pd.read_csv(input_path)

    # Combine fields into one text column
    print("Combining text fields...")
    df["text_raw"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["content"].fillna("")
    )

    print("Cleaning text (lowercase, remove punctuation, URLs)...")
    df["text_clean"] = df["text_raw"].apply(clean_text)

    print("Lemmatizing & removing stopwords (this may take a few minutes)...")
    df["text_lemma"] = df["text_clean"].apply(lemmatize_text)

    # Save processed dataset
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nPreprocessing complete!")
    print(f"Saved cleaned dataset → {output_path}")
    print(f"Rows processed: {len(df)}")


if __name__ == "__main__":
    main()
