import feedparser

GOOGLE_NEWS_URL = "https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"

def fetch_from_google_rss(keyword):
    """
    Fetch news articles from Google News RSS feed.
    Only provides title & summary (no full text).
    """
    url = GOOGLE_NEWS_URL.format(keyword=keyword.replace(" ", "+"))
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries:
        articles.append({
            "source": "google_rss",
            "keyword": keyword,
            "title": entry.get("title"),
            "description": entry.get("summary"),
            "content": None,                  # RSS does not provide full content
            "url": entry.get("link"),
            "published_at": entry.get("published"),
        })

    return articles
