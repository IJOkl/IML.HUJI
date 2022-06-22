from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    training_errors = np.zeros(cv)
    validation_errors = np.zeros(cv)
    x_folds, y_folds = np.array_split(X, cv), np.array_split(y, cv)
    for i in range(cv):
        i_x_fold = np.concatenate(x_folds[:i] + x_folds[i+1:])
        i_y_fold = np.concatenate(y_folds[:i] + y_folds[i+1:])
        estimator.fit(i_x_fold, i_y_fold)
        training_errors[i] = scoring(i_y_fold,estimator.predict(i_x_fold))
        validation_errors[i] = scoring(y_folds[i],estimator.predict(x_folds[i]))
    t_err ,v_err = training_errors.mean(),validation_errors.mean()
    return t_err, v_err


