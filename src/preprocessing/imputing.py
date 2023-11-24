import pandas as pd
from typing import Optional
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer


class AWSimpleImputer(TransformerMixin):

    def __init__(self, in_columns=None, out_columns=None, **kwargs):
        self.in_columns = in_columns
        self.out_columns = out_columns
        self.imputer = SimpleImputer(**kwargs)
        TransformerMixin.__init__(self, **kwargs)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        # --- Run imputation

        # Select appropriate columns
        if (self.in_columns is None) or not isinstance(self.in_columns, list):
            self.in_columns = X.columns

        # Select appropriate output
        if (self.out_columns is None) or not isinstance(self.out_columns, list):
            self.out_col = self.in_columns

        # Fit imputer
        X_i = X.loc[:, self.in_columns]
        self.imputer.fit(X_i, y, **kwargs)
        # Do imputation
        X, y = self.transform(X_i, y, **kwargs)
        return X, y

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        return self.fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        # --- Run transformation

        # Do imputation
        X_i = X.loc[:, self.in_columns]
        X_i = self.imputer.transform(X_i)

        # Make output
        X.loc[:, self.out_col] = X_i
        return X, y


if __name__ == '__main__':
    # --- TEST

    # Load data files
    PATH_TO_DATA = 'c:/Users/you.zerouali/Documents/Code/classif-factures/data'
    ls_files = [f'factures_{2010+i_}.xlsx' for i_ in range(5,6)]
    dfs = [pd.read_excel( "/".join([PATH_TO_DATA, 'Completes', i_]), 'DATA') for i_ in ls_files]
    df = pd.concat(dfs, axis=0).reset_index()

    # Make imputation
    imp = AWSimpleImputer(in_columns=['Montant_Ligne_Facture'])
    df_, _ = imp.fit_transform(df)
    assert df_['Montant_Ligne_Facture'].isnull().sum() == 0

    #Make sure it integrates in block
    from src.templates.blocks import AWBPipeline
    pipe = AWBPipeline([
        ('TEST', AWSimpleImputer(in_columns=['Montant_Ligne_Facture']))
    ])
    X_, y_ = pipe.fit_transform(df)
    assert df_['Montant_Ligne_Facture'].isnull().sum() == 0