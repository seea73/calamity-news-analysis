import requests

API_KEY = "c8431403ac5e46e2b1e40c750bd22243"
BASE_URL = "https://newsapi.org/v2/everything"

def fetch_from_newsapi(keyword):
    """
    Fetch articles from NewsAPI using keyword search.
    Returns a list of standardized article dictionaries.
    """
    params = {
        "q": keyword,
        "language": "en",
        "pageSize": 100,
        "sortBy": "publishedAt",
        "apiKey": API_KEY
    }

    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if data.get("status") != "ok":
            print(f"NewsAPI Error: {data.get('message')}")
            return []

        articles = []
        for art in data.get("articles", []):
            articles.append({
                "source": "newsapi",
                "keyword": keyword,
                "title": art.get("title"),
                "description": art.get("description"),
                "content": art.get("content"),
                "url": art.get("url"),
                "published_at": art.get("publishedAt"),
            })

        return articles

    except Exception as e:
        print("NewsAPI fetch failed:", e)
        return []
