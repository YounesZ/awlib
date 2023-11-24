import re
import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from ...submodules.awlib.src.text.common import STOPW
from ...submodules.awlib.src.text.cleaning import remove_accents, remove_digits, remove_punctuation, remove_stopwords


class TextCleaner(TransformerMixin):

    def __init__(self, in_columns=None, out_column='clean', max_digits=1, min_charac=1, remove_stopwords=True):
        self.in_columns = in_columns
        self.out_column = out_column
        self.max_digits = max_digits
        self.min_charac = min_charac
        self.remove_stopwords = remove_stopwords
        TransformerMixin.__init__(self)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
        # --- Run preprocessing

        # 1) Agregate columns
        if self.in_columns is None:
            self.in_columns = [X.columns[0]]

        df = X.copy()
        if len(self.in_columns) > 1:
            df[self.out_column] = X[self.in_columns].astype(str).agg(' '.join, axis=1)
        else:
            df[self.out_column] = df[self.in_columns]

        # 2) Remove punctuation
        df[self.out_column] = df[self.out_column].apply(remove_punctuation)
        # 3) Remove accents
        df[self.out_column] = df[self.out_column].apply(remove_accents)
        # 4) Remove non characters
        df[self.out_column] = df[self.out_column].apply(lambda x: re.sub('[^0-9a-zA-Z ]', '', x))
        # 5) Lower strings
        df[self.out_column] = df[self.out_column].apply(lambda x: x.lower())
        # 6) Tokenize sentences
        df[self.out_column] = df[self.out_column].apply(lambda x: x.split())
        # 7) Remove stopwords
        if self.remove_stopwords:
            # with open( path.join(ROOT_STOP, 'stopwords_fr.txt') ) as f:
            #    STOPW = f.readlines()[0].replace(' ', '').split(',')
            df[self.out_column] = df[self.out_column].apply(remove_stopwords, args=(STOPW,))
        # 8) Remove digits
        df[self.out_column] = df[self.out_column].apply(remove_digits, args=(self.min_charac, self.max_digits,))
        # 9) Re-join sentences
        df[self.out_column] = df[self.out_column].apply(lambda x: ' '.join(x))
        return df, y

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
        return self.fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
        return self.fit(X, y, **kwargs)