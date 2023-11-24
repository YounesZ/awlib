import re
import numpy as np
import pandas as pd
from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from submodules.awpipes.submodules.awlib.src.text.search import match_patterns_to_iterable


class DataSplitter(TransformerMixin):

    def __init__(self, test_size=0.2, shuffle=True, random_state=42):
        self.is_fit = False
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        TransformerMixin.__init__(self)

    def fit(self, X: Union[pd.DataFrame, np.array],
            y: Union[pd.DataFrame, np.array] = None, **kwargs) -> pd.DataFrame:
        # SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
        if y is None:
            y_train, y_test = None, None
            X_train, X_test = train_test_split(
                X,
                test_size=self.test_size,
                shuffle=self.shuffle,
                random_state=self.random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                shuffle=self.shuffle,
                random_state=self.random_state)

        # Store test split
        self.X_test = X_test
        self.y_test = y_test
        return X_train, y_train

    def transform(self, X: Union[pd.DataFrame, np.array] = None,
                  y: Union[pd.DataFrame, np.array] = None, **kwargs) -> pd.DataFrame:
        # Return test split for inference
        if (X is not None):
            return X, y
        else:
            return self.X_test, self.y_test

    def fit_transform(self, X: Union[pd.DataFrame, np.array],
                      y: Union[pd.DataFrame, np.array], **kwargs) -> pd.DataFrame:
        return self.fit(X, y, *kwargs)


class AWBFeatureLabelSplitter(TransformerMixin):

    def __init__(self, features=[], labels=[]):
        assert isinstance(features, list)
        assert isinstance(labels, list)

        self.features = features
        self.labels = labels

    def fit(self, X: Union[pd.DataFrame, np.array],
            y: Union[pd.DataFrame, np.array] = None, **kwargs) -> pd.DataFrame:

        # Select features from X
        FT = match_patterns_to_iterable(self.features, X.columns)
        X_ = X[FT]

        # Select labels from X
        LB = match_patterns_to_iterable(self.labels, X.columns)
        y_ = X[LB]
        return X_, y_

    def transform(self, X: Union[pd.DataFrame, np.array],
                  y: Union[pd.DataFrame, np.array] = None, **kwargs) -> pd.DataFrame:
        return self.fit(X, y, **kwargs)

    def fit_transform(self, X: Union[pd.DataFrame, np.array],
                      y: Union[pd.DataFrame, np.array] = None, **kwargs) -> pd.DataFrame:
        return self.fit(X, y, **kwargs)