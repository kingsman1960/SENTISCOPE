from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from flair.nn import Classifier
from flair.data import Sentence

nltk.download('vader_lexicon')

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