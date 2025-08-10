# Create Custom Transformer to count number of times a character appears
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CharCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, char):
        self.char = char

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([str(text).count(self.char) for text in X]).reshape(-1, 1)
        # return [[text.count(self.char) for text in X]]


class SpacyLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lemmatized = [
            ' '.join(
                token.lemma_ for token in doc
                if not token.is_stop
            )
            for doc in self.nlp.pipe(X)
        ]
        return lemmatized