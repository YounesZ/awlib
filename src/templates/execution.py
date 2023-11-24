# Imports
from copy import deepcopy
from sklearn.base import TransformerMixin, ClassifierMixin, RegressorMixin, BaseEstimator
from .io import get_transformer_outputs


def standard_obj_type(obj):

    # Determine object type
    if TransformerMixin in obj.__class__.__mro__:
        out = 'transformer'

    elif (ClassifierMixin in obj.__class__.__mro__) or (RegressorMixin in obj.__class__.__mro__):
        out = 'estimator'

    elif BaseEstimator in obj.__class__.__mro__:
        out = 'transformer'

    else:
        raise ValueError(f'Pipeline step {obj} is neither a transformer nor a model')

    return out


def standard_block_execution(obj, X, y, do_fit, **kwargs):
    # Determine block outputs
    y_ = deepcopy(y)
    X_ = deepcopy(X)

    obj_t = standard_obj_type(obj)

    if obj_t == 'transformer':  # Always output X
        # Determine N_OUT
        n_out = get_transformer_outputs(obj)
        if n_out not in [1, 2]:
            raise (ValueError, f'Incorrect number of outputs from transformer {obj.__class__}')

        if do_fit:
            if n_out == 1:
                X_ = obj.fit_transform(X_, **kwargs)
            elif n_out == 2:
                X_, y_ = obj.fit_transform(X_, y_, **kwargs)
        else:
            if n_out == 1:
                X_ = obj.transform(X_)
            elif n_out == 2:
                X_, y_ = obj.transform(X_, y_, **kwargs)

    elif obj_t == 'estimator':
        n_out = 1  # Estimators only ouput y
        if do_fit:
            obj.fit(X, y, **kwargs)
        X_, y_ = y_, obj.predict(X_)

    return X_, y_
