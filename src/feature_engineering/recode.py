import numpy as np
import pandas as pd

from typing import Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class AWLabelEncoder(TransformerMixin):
    def __init__(self):
        self.is_fit = False
        self.encoder= LabelEncoder()
        TransformerMixin.__init__(self)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
        # SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
        self.encoder.fit(y)
        y_encoded = self.encoder.transform(y)
        return X, y_encoded

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
        # Filter labels
        filter_out  = np.ones(y.shape).astype(bool)
        for i_ in y.drop_duplicates().values.flatten():
            try:
                self.encoder.transform([i_])
            except:
                filter_out[ y==i_] = False
                print(f'Unknown class {i_}')

        # TODO: MODIFY THE WAY MISSING CLASSES ARE HANDLED - RECODE WITH -1 INSTEAD OF DROPPING

        # Make filter
        X_ = X[filter_out]
        y_ = self.encoder.transform(y[filter_out])
        return X_, y_

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame], **kwargs) -> pd.DataFrame:
        return self.fit(X, y, *kwargs)


class AWFeatureEncoder(TransformerMixin):

    def __init__(self, in_columns=[], out_prefix=[], encoder=OneHotEncoder(), dropin=True):
        # Output prefix
        self.out_prefix = []
        if isinstance(out_prefix, list) and (len(out_prefix)>0):
            self.out_prefix = out_prefix
        # Encoders
        self.encoders = []
        if isinstance(in_columns, list) and (len(in_columns)>0):
            enc_params = encoder.get_params()
            self.encoders = [encoder.__class__(**enc_params) for i_ in in_columns]
        self.dropin = dropin
        self.in_columns = in_columns
        TransformerMixin.__init__(self)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        # Loop on input columns
        for h_, i_ in zip(self.encoders, self.in_columns):

            # Isolate column - without nans
            i_col = X.loc[~X[i_].isna(), i_]
            i_col_exp = np.expand_dims(i_col, axis=1)
            # Encode
            h_.fit(i_col_exp)

        # Transform
        X, y = self.transform(X, y, **kwargs)
        return X, y

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        # Loop on input columns
        for h_, i_, k_ in zip(self.encoders, self.in_columns, self.out_prefix):

            # Isolate column - without nans
            i_col = X.loc[~X[i_].isna(), i_]
            i_col_exp = np.expand_dims(i_col, axis=1)
            # Encode
            i_col_enc = h_.transform(i_col_exp).toarray()

            # Put in new df
            nCols = i_col_enc.shape[1]
            newDf = pd.DataFrame(data=np.zeros([len(X), nCols]),
                                 columns=[f'{k_}_{j_}' for j_ in range(nCols)],
                                 index=X.index)
            newDf.loc[i_col.index,:] = i_col_enc

            # Append new df to X
            X = pd.concat([X, newDf], axis=1)
        return X, y

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        return self.fit(X, y, **kwargs)


class AWBShapeAs2D(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X: np.array, y: np.array, **kwargs) -> np.array:
        y_ = y
        if len(np.shape(y)) == 1:
            y_ = np.reshape(y, [-1, 1])

        return X, y_

    def transform(self, X: np.array, y: np.array, **kwargs) -> np.array:
        return self.fit(X, y, **kwargs)

    def fit_transform(self, X: np.array, y: np.array, **kwargs) -> np.array:
        return self.fit(X, y, **kwargs)



if __name__ == '__main__':

    # ------------- #
    # --- TESTS --- #
    # ------------- #

    from config import PATHS
    from submodules.awpipes.src.templates.blocks import AWBPipeline

    DATA_FILE = ['factures_2015.xlsx']
    df = [pd.read_excel("/".join([PATHS['ROOT_DATA'], "Completes", i_]), 'DATA') for i_ in DATA_FILE]
    df = pd.concat(df, axis=0).reset_index()

    # --- FEATURE ENCODING --- #
    # Apply feature encoding
    enc = AWFeatureEncoder(in_columns=['key01', 'key03'],
                           out_prefix=['key01_cat', 'key03_cat'],
                           dropin=True)
    X, y = enc.fit_transform(df)


    # Encapsulate in a pipeline
    pipe = AWBPipeline([
        ('Categorical encoder | adds columns to the features table', AWFeatureEncoder(in_columns=['key01', 'key03'],
                                                                                      out_prefix=['key01_recoded', 'key03_recoded'],
                                                                                      dropin=True))
    ])
    X_, y_ = pipe.fit_transform(df)
    X__, y__ = pipe.transform(df)
