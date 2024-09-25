import requests
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from flair.nn import Classifier
from flair.data import Sentence

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
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


class EntityExtractor:
    def __init__(self):
        self.flair_ner_model = Classifier.load('ner')

    def extract_entities_flair(self, text):
        sentence = Sentence(text)
        self.flair_ner_model.predict(sentence)
        return [(entity.text, entity.tag) for entity in sentence.get_spans('ner')]

    def extract_entities_nltk(self, text):
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        tree = nltk.ne_chunk(pos_tags)
        entities = []
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                entity_text = ' '.join([word for word, tag in subtree.leaves()])
                entity_label = subtree.label()
                entities.append((entity_text, entity_label))
        return entities


class SentimentAnalyzer:
    def __init__(self):
        self.finbert_tokenizer, self.finbert_model = self._load_finbert()
        self.esgbert_tokenizer, self.esgbert_model = self._load_esgbert()
        self.finbert_tone_tokenizer, self.finbert_tone_model = self._load_finbert_tone()
        self.flair_sentiment_model = Classifier.load('en-sentiment')

    def _load_finbert(self):
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return tokenizer, model

    def _load_esgbert(self):
        tokenizer = AutoTokenizer.from_pretrained("nbroad/ESG-BERT")
        model = AutoModelForSequenceClassification.from_pretrained("nbroad/ESG-BERT")
        return tokenizer, model

    def _load_finbert_tone(self):
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        return tokenizer, model

    def analyze_sentiment_finbert(self, text):
        inputs = self.finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment_scores = probabilities[0].tolist()
        labels = ['Negative', 'Neutral', 'Positive']
        return {label: score for label, score in zip(labels, sentiment_scores)}

    def analyze_sentiment_vader(self, text):
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)

    def analyze_sentiment_esgbert(self, text):
        inputs = self.esgbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.esgbert_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment_scores = probabilities[0].tolist()
        labels = ['Negative', 'Neutral', 'Positive']
        return {label: score for label, score in zip(labels, sentiment_scores)}

    def analyze_sentiment_finbert_tone(self, text):
        inputs = self.finbert_tone_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.finbert_tone_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment_scores = probabilities[0].tolist()
        labels = ['Negative', 'Neutral', 'Positive']
        return {label: score for label, score in zip(labels, sentiment_scores)}

    def analyze_sentiment_flair(self, text):
        sentence = Sentence(text)
        self.flair_sentiment_model.predict(sentence)
        return {'sentiment': sentence.labels[0].value, 'score': sentence.labels[0].score}