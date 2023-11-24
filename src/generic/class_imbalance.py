# Imports
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from scipy.sparse.csr import csr_matrix
from ...submodules.awlib.src.generic.array import replicate_array, shuffle_pair


class SMOTE(TransformerMixin):

    def __init__(self, random_state=42):
        self.is_fit = False
        self.random_state = random_state
        TransformerMixin.__init__(self)

    def fit(self, X: np.array,
            y: np.array = None, **kwargs) -> np.array:
        assert isinstance(y, np.ndarray)

        self.classes = np.unique(y)
        self.maxC = 0
        if len(y ) >0:
            self.maxC = max(np.histogram(y, len(self.classes))[0])
        self.is_fit = True

        # Ensure input has the right format
        if isinstance(X, csr_matrix):
            X = X.toarray()
        if isinstance(y, csr_matrix):
            y = y.toarray()

        assert type(X) in [np.ndarray, csr_matrix]
        assert type(y) in [np.ndarray, csr_matrix]

        # Loop on classes
        nX, nY  = [], []
        for x_c, i_c in enumerate(self.classes):
            # Slice arrays
            x_i = X[ y==i_c]
            y_i = y[ y==i_c]

            # Replicate array
            nX.append(replicate_array(x_i, self.maxC))
            nY.append(replicate_array(y_i, self.maxC) )

        # Contactenate
        if len(self.classes ) >0:
            nX = np.concatenate(nX, axis=0)
            nY = np.concatenate(nY)

            # Shuffle
            nX, nY = shuffle_pair(nX, nY, self.random_state)
        else:
            nX, nY = np.array([]), np.array([])
        return nX, nY

    def transform(self, X: np.array,
                  y: np.array = None, **kwargs) -> np.array:
        # Pass-through function - we don't oversample for inference
        return X, y

    def fit_transform(self, X: np.array,
                      y: np.array = None, **kwargs) -> np.array:
        X, y = self.fit(X, y, **kwargs)
        return X, y


