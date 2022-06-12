from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class Vectorizer(TransformerMixin):
    def __init__(self,
                 vectorizer='count',
                 binary=False,
                 min_df=5):
        self.binary = binary
        self.min_df = min_df
        self.vectorizer = vectorizer
        
        if vectorizer == 'count':
            vect = CountVectorizer(min_df=min_df,binary=binary)
        elif vectorizer == 'tfidf':
            vect = TfidfVectorizer(min_df=min_df)
        
        self.vect = vect

    def fit(self,X_tr):
        self.vect.fit(X_tr)
        return self
    
    def transform(self,X):
        result = self.vect.transform(X)
        return result

    def get_feature_name(self,prefix=''):
        binary = '_binary' if self.binary else ''
        fname = f"{prefix}_{self.vectorizer}_{self.min_df}{binary}.npy"
        return fname

class TopicTransformer(Vectorizer):
    def __init__(self,
                vectorizer='count',
                binary=False,
                n_components=None,
                min_df=5):
        super().__init__(vectorizer=vectorizer,
                         binary=binary,
                         min_df=min_df)
        self.n_components = n_components
        
        assert n_components is not None,'when use_topic activated,n_components must not be None'
        topic_m = TruncatedSVD(n_components=n_components)
        self.topic_m = topic_m
    
    def fit(self,X_tr):
        self.topic_m.fit(X_tr)
        return self
    
    def transform(self,X):
        result = self.topic_m.transform(X)
        return result

    def get_feature_name(self,prefix=''):
        feature_name = super().get_feature_name(prefix=prefix)
        n_components = '_' + str(self.n_components)
        pos = feature_name.find('.')
        fname = feature_name[:pos]
        fname = fname + n_components + '.npy'
        return fname