import os
import pandas as pd
import matplotlib.pyplot as plt

def main():

    # ------------------------------------------------------
    # Paths
    # ------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR, "data", "processed", "news_with_countries.csv")
    results_dir = os.path.join(BASE_DIR, "results", "figures")
    tables_dir = os.path.join(BASE_DIR, "results", "tables")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("Loading dataset:", input_path)
    df = pd.read_csv(input_path)

    # ------------------------------------------------------
    # Clean location names (remove weird entries like "AI")
    # Keep only alphabetic names with length > 2
    # ------------------------------------------------------
    df["location"] = df["location"].fillna("").astype(str)
    df = df[df["location"].str.match(r"^[A-Za-z ].{2,}$")]   # remove 1-2 char junk like "AI"

    # ------------------------------------------------------
    # Country counts per calamity
    # ------------------------------------------------------
    print("Aggregating counts...")

    grouped = (
        df.groupby(["calamity_label", "location"])
        .size()
        .reset_index(name="count")
    )

    # Save summary table
    grouped.to_csv(
        os.path.join(tables_dir, "country_counts.csv"),
        index=False
    )

    # All calamities
    calamities = sorted(df["calamity_label"].unique())

    # ------------------------------------------------------
    # 1️⃣ TOP COUNTRIES BY CALAMITY — Bar charts
    # ------------------------------------------------------
    for calamity in calamities:
        sub = grouped[grouped["calamity_label"] == calamity] \
            .sort_values("count", ascending=False) \
            .head(10)

        plt.figure(figsize=(9, 5))
        plt.bar(sub["location"], sub["count"], color="skyblue")
        plt.title(f"Top 10 Countries Mentioned in {calamity.capitalize()} News")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        fig_path = os.path.join(results_dir, f"top10_{calamity}.png")
        plt.savefig(fig_path)
        plt.close()

        print("Saved:", fig_path)

    # ------------------------------------------------------
    # 2️⃣ HEATMAP (country × calamity)
    # ------------------------------------------------------
    print("Creating heatmap...")
    heatmap_df = grouped.pivot(
        index="location",
        columns="calamity_label",
        values="count"
    ).fillna(0)

    heatmap_csv_path = os.path.join(tables_dir, "heatmap_data.csv")
    heatmap_df.to_csv(heatmap_csv_path)

    # Plot heatmap using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_df, aspect="auto", cmap="Blues")
    plt.title("Calamity vs Country Heatmap")
    plt.xlabel("Calamity Type")
    plt.ylabel("Country")

    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=45)
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)

    # Add count text inside boxes
    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            plt.text(j, i, int(heatmap_df.iloc[i, j]),
                     ha="center", va="center", fontsize=7)

    plt.tight_layout()

    heatmap_fig_path = os.path.join(results_dir, "heatmap_calamity_country.png")
    plt.savefig(heatmap_fig_path)
    plt.close()

    print("Saved:", heatmap_fig_path)

    print("\nAll visualizations completed!")


if __name__ == "__main__":
    main()
