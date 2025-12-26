import os
import pandas as pd

# Allow imports from sources/ folder
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sources.newsapi_fetch import fetch_from_newsapi
from sources.google_rss_fetch import fetch_from_google_rss
from sources.gdelt_fetch import fetch_from_gdelt

CALAMITY_TERMS = ["flood", "earthquake", "wildfire", "hurricane", "drought"]

def main():
    all_articles = []

    for term in CALAMITY_TERMS:
        print(f"\nFetching articles for: {term}")

        # Each API called here
        newsapi_articles = fetch_from_newsapi(term)
        rss_articles = fetch_from_google_rss(term)
        gdelt_articles = fetch_from_gdelt(term)

        print(f" NewsAPI: {len(newsapi_articles)}")
        print(f" Google RSS: {len(rss_articles)}")
        print(f" GDELT: {len(gdelt_articles)}")

        all_articles.extend(newsapi_articles)
        all_articles.extend(rss_articles)
        all_articles.extend(gdelt_articles)

    df = pd.DataFrame(all_articles)

    # Remove exact duplicates
    df.drop_duplicates(subset=["url", "title"], inplace=True)

    # Ensure data folder exists
    os.makedirs("data/raw", exist_ok=True)

    # Save CSV
    df.to_csv("data/raw/all_news_raw.csv", index=False)

    print(f"\nSaved {len(df)} total unique articles → data/raw/all_news_raw.csv")

if __name__ == "__main__":
    main()
