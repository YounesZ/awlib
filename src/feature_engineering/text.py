# Imports
import numpy as np
import pandas as pd

from typing import Union, Optional
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class AWBTfidfVectorizer(TransformerMixin):

    def __init__(self, in_columns=[], out_prefix=[], use_idf=True):
        self.in_columns = in_columns
        self.out_prefix = out_prefix
        # Instantiate vectorizers
        self.vectorizer = []
        for i_ in self.in_columns:
            self.vectorizer.append( TfidfVectorizer(use_idf=use_idf) )

    def fit(self, X: Union[pd.DataFrame],
            y: Union[pd.DataFrame, np.array] = None, **kwargs) -> pd.DataFrame:

        # Turn dataframe column to list
        for i_, j_ in zip(self.in_columns, self.vectorizer):
            # Extract + transform column
            i_col = X[i_]
            # Remove NaNs
            i_col_nan = i_col[~i_col.isna()]
            j_.fit(i_col_nan)

        # Call transform
        X, y = self.transform(X, y, **kwargs)
        return X, y

    def transform(self, X: Union[pd.DataFrame],
                  y: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:

        # Turn dataframe column to list
        for i_, j_,k_ in zip(self.in_columns, self.vectorizer, self.out_prefix):
            # Extract + transform column
            i_col = X[i_]
            # Remove NaNs
            i_col_nan = i_col[~i_col.isna()]
            o_col = j_.transform(i_col_nan).toarray()

            # Put in new df
            nCols = o_col.shape[1]
            newDf = pd.DataFrame(data=np.zeros([len(X), nCols]),
                                 columns=[f'{k_}_{j_}' for j_ in range(nCols)],
                                 index=X.index)
            newDf.loc[i_col_nan.index, :] = o_col

            # Append new df to X
            X = pd.concat([X, newDf], axis=1)

        return X, y

    def fit_transform(self, X: Union[pd.DataFrame, np.array],
                      y: Union[pd.DataFrame, np.array] = None, **kwargs) -> pd.DataFrame:
        return self.fit(X, y, **kwargs)


if __name__ == '__main__':

    # ------------- #
    # --- TESTS --- #
    # ------------- #

    from config import PATHS
    from submodules.awpipes.src.templates.blocks import AWBPipeline

    DATA_FILE = ['factures_2015.xlsx']
    df = [pd.read_excel("/".join([PATHS['ROOT_DATA'], "Completes", i_])) for i_ in DATA_FILE]
    df = pd.concat(df, axis=0).reset_index().iloc[:1000]

    # --- FEATURE ENCODING --- #
    # Apply feature encoding
    enc = AWBTfidfVectorizer(in_columns=['descp'],
                             out_prefix=['descp_vec'])
    X, y = enc.fit_transform(df)

    # Encapsulate in a pipeline
    pipe = AWBPipeline([
        ('Text vectorizer | standard SKlearn Tf-idf', AWBTfidfVectorizer(in_columns=['descp'],
                                                                         out_prefix=['descp_vec']))
    ])
    X_, y_ = pipe.fit_transform(df)
    X__, y__ = pipe.transform(df)