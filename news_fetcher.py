import requests

class NewsFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.trusted_sources = [
            'reuters.com', 'apnews.com', 'bbc.com', 'npr.org', 'wsj.com',
            'nytimes.com', 'washingtonpost.com', 'economist.com', 'ft.com',
            'bloomberg.com', 'cnbc.com', 'forbes.com', 'finance.yahoo.com'
        ]

    def fetch_financial_news(self, sector):
        financial_keywords = "finance OR market OR stock OR economy OR investment"
        query = f"{sector} AND ({financial_keywords})"
        domains = ','.join(self.trusted_sources)
        url = f"https://newsapi.org/v2/everything?q={query}&domains={domains}&apiKey={self.api_key}&language=en&sortBy=relevancy&pageSize=10"
        
        response = requests.get(url)
        data = response.json()
        return data.get('articles', [])