import re
import pandas as pd
from typing import Optional
from config import TEXT_CLEANING
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from submodules.awlib.src.text.cleaning import remove_accents, remove_digits, remove_punctuation, remove_exact_match, stem_string


class TextCleaner(TransformerMixin):

    def __init__(self, in_columns=None, out_column='clean', max_digits=1, min_charac=1, remove_stopwords=True, remove_names=True, remove_locations=True, remove_dates=True, custom_stemming=True):
        self.in_columns = in_columns
        self.out_column = out_column
        self.max_digits = max_digits
        self.min_charac = min_charac
        self.remove_stopwords = remove_stopwords
        self.remove_dates = remove_dates
        self.remove_names = remove_names
        self.remove_locations = remove_locations
        self.custom_stemming = custom_stemming
        TransformerMixin.__init__(self)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
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
        # 3) Lower strings
        df[self.out_column] = df[self.out_column].apply(lambda x: x.lower())
        # 4) Remove accents
        df[self.out_column] = df[self.out_column].apply(remove_accents)
        # 5) Remove non characters
        df[self.out_column] = df[self.out_column].apply(lambda x: re.sub('[^0-9a-zA-Z ]', '', x))
        # 6) Tokenize sentences
        df[self.out_column] = df[self.out_column].apply(lambda x: x.split())

        # 7) --- Remove non-contextual tokens
        # Stopwords
        if self.remove_stopwords:
            df[self.out_column] = df[self.out_column].apply(remove_exact_match, args=(TEXT_CLEANING['STOPWORDS'],))
        # Dates
        if self.remove_dates:
            df[self.out_column] = df[self.out_column].apply(remove_exact_match, args=(TEXT_CLEANING['DATES'],))
        # Names
        if self.remove_names:
            df[self.out_column] = df[self.out_column].apply(remove_exact_match, args=(TEXT_CLEANING['NAMES'],))
        # Locations
        if self.remove_locations:
            df[self.out_column] = df[self.out_column].apply(remove_exact_match, args=(TEXT_CLEANING['LOCATIONS'],))

        # 8) Custom stemming
        if self.custom_stemming:
            df[self.out_column] = df[self.out_column].apply(stem_string, args=(TEXT_CLEANING['STEMS'],))

        # 9) Remove digits
        df[self.out_column] = df[self.out_column].apply(remove_digits, args=(self.min_charac, self.max_digits,))
        # 10) Re-join sentences
        df[self.out_column] = df[self.out_column].apply(lambda x: ' '.join(x))
        return df, y

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        return self.fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        return self.fit(X, y, **kwargs)



if __name__ == '__main__':
    # ------------- #
    # --- TESTS --- #
    # ------------- #

    from config import PATHS
    from src.templates.blocks import AWBPipeline

    DATA_FILE = ['factures_2015.xlsx']
    df = [pd.read_excel("/".join([PATHS['ROOT_DATA'], "Completes", i_])) for i_ in DATA_FILE]
    df = pd.concat(df, axis=0).reset_index()

    # --- FEATURE ENCODING --- #
    # Apply feature encoding
    tcl = TextCleaner(in_columns=['Objet_Facture', 'descp'],
                      out_column='clean_text',
                      max_digits=1,
                      min_charac=1,
                      remove_stopwords=True,
                      remove_names=True,
                      remove_locations=True,
                      remove_dates=True)
    X, y = tcl.fit_transform(df)

    # Encapsulate in a pipeline
    pipe = AWBPipeline([
        ('Text vectorizer | standard SKlearn Tf-idf', tcl)
    ])
    X_, y_ = pipe.fit_transform(df)
    X__, y__ = pipe.transform(df)