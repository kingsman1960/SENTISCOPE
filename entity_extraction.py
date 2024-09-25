import nltk
from flair.nn import Classifier
from flair.data import Sentence

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

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