from sklearn.pipeline import Pipeline
from .execution import standard_block_execution


class AWBPipeline(Pipeline):

    def __init__(self, steps, **kwargs):
        return super().__init__(steps, **kwargs)

    def fit(self, X, y=None, **fit_params):
        print('\nFitting ...')
        for x_ ,(i_, j_) in enumerate(self.steps):

            print(f'\t{i_}')
            X, y = standard_block_execution(j_, X, y, True, **fit_params)

        print('done.\n')
        return X, y

    def transform(self, X, y=None, **fit_params):
        print('\nTransforming ...')
        for x_ ,(i_, j_) in enumerate(self.steps):

            print(f'\t{i_}')
            X, y = standard_block_execution(j_, X, y, False, **fit_params)

        print('done.\n')

        return X, y

    def predict(self, X, y=None, **fit_params):
        return self.transform(X, y, **fit_params)

    def fit_transform(self, X, y=None, **fit_params):
        print('Called fit_transform')
        return self.fit(X, y, **fit_params)