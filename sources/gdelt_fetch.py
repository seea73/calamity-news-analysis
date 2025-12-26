import requests, json

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def fetch_from_gdelt(keyword):
    params = {
        "query": keyword,
        "mode": "ArtList",
        "maxrecords": 50,
        "format": "json"
    }

    try:
        response = requests.get(GDELT_URL, params=params, timeout=10)

        if response.status_code != 200:
            print("GDELT Request failed:", response.status_code)
            return []

        # GDELT sometimes returns blank → avoid crash
        if not response.text.strip():
            print("GDELT returned empty response.")
            return []

        try:
            data = response.json()
        except json.decoder.JSONDecodeError:
            print("GDELT returned invalid JSON.")
            return []

        if "articles" not in data:
            return []

        articles = []
        for art in data["articles"]:
            articles.append({
                "source": "gdelt",
                "keyword": keyword,
                "title": art.get("title"),
                "description": art.get("sourcecountry"),
                "content": None,
                "url": art.get("url"),
                "published_at": art.get("seendate"),
            })

        return articles

    except Exception as e:
        print("GDELT fetch failed:", e)
        return []
