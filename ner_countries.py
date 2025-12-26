import os
import pandas as pd
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# -------------------------------------------------
# Extract GPE (countries, cities) + LOC (locations)
# -------------------------------------------------
def extract_locations(text):
    if not isinstance(text, str):
        return []

    doc = nlp(text)

    # CORRECT: use doc.ents, not doc
    locations = [
        ent.text
        for ent in doc.ents
        if ent.label_ in ("GPE", "LOC")   # Named entities only
    ]

    # Deduplicate
    return list(set(locations))


# -------------------------------------------------
# Country normalization (optional improvements)
# -------------------------------------------------
COUNTRY_MAP = {
    "U.S.": "United States",
    "US": "United States",
    "USA": "United States",
    "U.K.": "United Kingdom",
    "UK": "United Kingdom",
    "England": "United Kingdom",
    "Russia": "Russian Federation",
}

def normalize_country(name):
    return COUNTRY_MAP.get(name, name)


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(BASE_DIR, "data", "processed", "news_labeled.csv")
    output_path = os.path.join(BASE_DIR, "data", "processed", "news_with_countries.csv")

    print("Loading dataset:", input_path)
    df = pd.read_csv(input_path)

    print("Extracting countries & locations (this may take a few minutes)...")
    df["locations_raw"] = df["text_raw"].apply(extract_locations)

    print("Normalizing country names...")
    df["locations_norm"] = df["locations_raw"].apply(
        lambda lst: [normalize_country(x) for x in lst]
    )

    print("Expanding rows...")
    df_expanded = df.explode("locations_norm")
    df_expanded = df_expanded.rename(columns={"locations_norm": "location"})
    df_expanded = df_expanded.dropna(subset=["location"])

    df_expanded.to_csv(output_path, index=False)

    print("\nNER Extraction complete!")
    print("Saved →", output_path)

    print("\nTop 10 most common locations:")
    print(df_expanded["location"].value_counts().head(10))


if __name__ == "__main__":
    main()
