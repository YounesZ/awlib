# Imports
import numpy as np

from math import ceil
from sklearn.base import TransformerMixin
from scipy.sparse import csr_matrix
from submodules.awlib.src.generic.array import replicate_array, shuffle_pair


class SMOTE(TransformerMixin):

    def __init__(self, row_limit=None, random_state=42):
        self.is_fit = False
        self.row_limit = row_limit
        self.random_state = random_state
        TransformerMixin.__init__(self)

    def fit(self, X: np.array,
            y: np.array = None, **kwargs) -> np.array:
        assert isinstance(y, np.ndarray)

        self.classes = np.unique(y)
        self.maxC = 0
        if len(y)>0:
            self.maxC = max(np.histogram(y, len(self.classes))[0])
        self.is_fit = True

        # Check if restrictions if row table size
        if not self.row_limit is None:
            # Make sure table is big enough
            assert self.row_limit > len(self.classes)
            self.maxC = ceil(self.row_limit / len(self.classes))

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
            x_i = X[y==i_c]
            y_i = y[y==i_c]

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



if __name__ == '__main__':
    # ------------- #
    # --- TESTS --- #
    # ------------- #

    import pandas as pd
    from config import PATHS
    from src.templates.blocks import AWBPipeline
    from sklearn.preprocessing import LabelEncoder

    DATA_FILE = ['factures_2015.xlsx']
    df = [pd.read_excel("/".join([PATHS['ROOT_DATA'], "Completes", i_])) for i_ in DATA_FILE]
    df = pd.concat(df, axis=0).reset_index()

    # --- FEATURE ENCODING --- #
    # Apply feature encoding
    df.loc[df['Service ABC'].isna(), 'Service ABC'] = 'NaN'
    X, y = df[['Montant_Ligne_Facture', 'descp']], df['Service ABC']
    smt = SMOTE(row_limit=100000)
    enc = LabelEncoder()
    X_, y_ = smt.fit_transform(X.values,
                               enc.fit_transform(y))

    # Encapsulate in a pipeline
    pipe = AWBPipeline([
        ('SMOTE | custom class balancer', smt)
    ])
    X_, y_ = pipe.fit_transform(X.values, enc.fit_transform(y))
    X__, y__ = pipe.transform(X.values, enc.fit_transform(y))