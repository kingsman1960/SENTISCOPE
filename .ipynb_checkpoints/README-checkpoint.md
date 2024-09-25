# Sentiscope

**Sentiscope** is a powerful Python package for analyzing sentiment and extracting entities from financial news articles. It leverages multiple state-of-the-art sentiment analysis models and named entity recognition techniques to provide comprehensive insights into financial news.

## Features

- Fetch financial news articles for specific sectors using NewsAPI
- Perform sentiment analysis using multiple models:
  - FinBERT
  - VADER
  - ESG-BERT
  - FinBERT-Tone
  - Flair
- Extract named entities using:
  - NLTK
  - Flair

## Installation
```python
pip install sentiscope
```

## Quick Start

```python
from sentiscope import SentimentAnalyzer, NewsFetcher, EntityExtractor

# Initialize components
sentiment_analyzer = SentimentAnalyzer()
news_fetcher = NewsFetcher(api_key="your_newsapi_key_here")
entity_extractor = EntityExtractor()

# Fetch news
news = news_fetcher.fetch_financial_news("Banking")

# Analyze sentiment and extract entities
for article in news:
    finbert_sentiment = sentiment_analyzer.analyze_sentiment_finbert(article['description'])
    vader_sentiment = sentiment_analyzer.analyze_sentiment_vader(article['description'])
    entities = entity_extractor.extract_entities_flair(article['description'])
    
    print(f"Title: {article['title']}")
    print(f"FinBERT Sentiment: {finbert_sentiment}")
    print(f"VADER Sentiment: {vader_sentiment}")
    print(f"Entities: {entities}")
    print("---")
```

## Detailed Usage
### Sentiment Analysis
Sentiscope offers multiple sentiment analysis models:
```python
text = "The company reported strong earnings, beating market expectations."

finbert_sentiment = sentiment_analyzer.analyze_sentiment_finbert(text)
vader_sentiment = sentiment_analyzer.analyze_sentiment_vader(text)
esgbert_sentiment = sentiment_analyzer.analyze_sentiment_esgbert(text)
finbert_tone_sentiment = sentiment_analyzer.analyze_sentiment_finbert_tone(text)
flair_sentiment = sentiment_analyzer.analyze_sentiment_flair(text)
```

### News Fetching
Fetch financial news for specific sectors:
```python
news = news_fetcher.fetch_financial_news("Technology")
```
## Models
**Sentiscope** uses the following pre-trained models:
- FinBERT: Fine-tuned BERT for financial sentiment analysis
- ESG-BERT: BERT model for environmental, social, and governance (ESG) sentiment
- FinBERT-Tone: BERT model for financial sentiment tone analysis
- Flair: General-purpose sentiment analysis and named entity recognition
- VADER: Rule-based sentiment analysis tool

## Dependencies
- transformers
- torch
- nltk
- flair
- requests

## Note
This package requires an API key from NewsAPI to fetch financial news articles. You can obtain a key by registering at https://newsapi.org.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
## Support
If you encounter any issues or have questions, please file an issue on the GitHub repository.
## Acknowledgements
Sentiscope is built on top of several open-source projects and pre-trained models. We are grateful to the developers and researchers who have made their work available to the community.
