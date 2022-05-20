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
    train_scores = np.zeros(cv)
    validation_scores = np.zeros(cv)
    folds = np.remainder(np.arange(X.shape[0]),cv)
    for k in range(cv):
        exclude = k != folds
        train_X, train_y = X[exclude], y[exclude]
        test_X ,test_y = X[~exclude], y[~exclude]
        estimator.fit(train_X,train_y)
        validation_scores[k] = scoring(test_y,estimator.predict(test_X),None)
        train_scores[k] = scoring(train_y,estimator.predict(train_X),None)
    train_score = np.mean(train_scores)
    validation_score = np.mean(validation_scores)
    return train_score,validation_score


