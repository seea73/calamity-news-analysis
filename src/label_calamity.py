import os
import pandas as pd

# ---------------------------------------
# Dictionary of calamity keywords
# ---------------------------------------
CALAMITY_KEYWORDS = {
    "flood": ["flood", "flooding", "heavy rain", "river overflow"],
    "earthquake": ["earthquake", "seismic", "tremor", "aftershock"],
    "wildfire": ["wildfire", "forest fire", "bushfire"],
    "hurricane": ["hurricane", "cyclone", "typhoon", "storm"],
    "drought": ["drought", "dry spell", "water shortage"]
}


# ---------------------------------------
# Label assignment function
# ---------------------------------------
def assign_label(text):
    text = text.lower()
    scores = {c: 0 for c in CALAMITY_KEYWORDS}

    # Count keyword occurrences for each calamity
    for calamity, words in CALAMITY_KEYWORDS.items():
        for w in words:
            if w in text:
                scores[calamity] += 1

    # Pick calamity with highest score
    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "unknown"
    return best


# ---------------------------------------
# Main pipeline
# ---------------------------------------
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "processed", "news_clean.csv")
    output_path = os.path.join(base_dir, "data", "processed", "news_labeled.csv")

    print("Loading:", input_path)
    df = pd.read_csv(input_path)

    # Use the clean text for keyword matching
    print("Assigning calamity labels...")
    df["calamity_label"] = df["text_clean"].astype(str).apply(assign_label)

    # Save output
    df.to_csv(output_path, index=False)
    print("\nLabeling complete!")
    print("Saved →", output_path)

    print("\nLabel distribution:")
    print(df["calamity_label"].value_counts())


if __name__ == "__main__":
    main()
