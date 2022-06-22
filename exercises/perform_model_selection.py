from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.linspace(-1.2, 2,n_samples)
    y = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y_noise = y + np.random.normal(0, noise, n_samples)
    x_train, y_train, x_test, y_test = split_train_test(pd.DataFrame(x), pd.Series(y_noise), 2 / 3)
    x_train, y_train = x_train.to_numpy()[:, 0], y_train.to_numpy()
    x_test, y_test = x_test.to_numpy()[:, 0], y_test.to_numpy()
    plt.scatter(x_train, y_train, label='Train Set')
    plt.scatter(x_test, y_test, label='Test Set')
    plt.scatter(x, y, label='Noise-less model', s=3, c='green')
    plt.title(f'Samples drawn from polynomial model with noise of {noise}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    validation_err = []
    training_err = []
    for k in range(11):
        model = PolynomialFitting(k)
        t_err,v_err = cross_validate(model,x_train,y_train,mean_square_error)
        validation_err.append(v_err)
        training_err.append(t_err)
    plt.plot([k for k in range(11)],training_err,label = "Training Error")
    plt.plot([k for k in range(11)], validation_err, label="Validation Error")
    plt.xlabel('Polynom degree')
    plt.ylabel('Errors')
    plt.title('Training and Validation Error as a function on Polynom degree')
    plt.legend()
    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(validation_err)
    k_star_model = PolynomialFitting(k_star).fit(x_train,y_train)
    test_err = round(mean_square_error(y_test, k_star_model.predict(x_test)), 2)

    print(f"*******Q3 with {noise} Noise  best k is: {k_star} ********")
    print(f"test error is: {round(test_err,2)}")
    print(f"validation error is: {round(validation_err[k_star],2)}")
    print("***********************************************")




def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions

    X,y = datasets.load_diabetes(return_X_y=True,as_frame=True)
    x_train,y_train, test_x,test_y = X[:50].values, y[:50].values,X[50:].values,y[50:].values
    # x_train, y_train = x_train.to_numpy(), y_train.to_numpy()
    # test_x, test_y = test_x.to_numpy(), test_y.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    regs = np.linspace(0.0001, 1, n_evaluations)  # check different ranges
    r_training_err = []
    r_validation_err = []
    l_training_err = []
    l_validation_err = []

    for i in range(n_evaluations):
        r_model = RidgeRegression(regs[i])
        r_t_err, r_v_err = cross_validate(r_model, x_train, y_train, mean_square_error)
        r_training_err.append(r_t_err)
        r_validation_err.append(r_v_err)

        l_model = Lasso(regs[i])
        l_t_err, l_v_err = cross_validate(l_model, x_train, y_train, mean_square_error)
        l_training_err.append(l_t_err)
        l_validation_err.append(l_v_err)
    go.Figure([go.Scatter(y=r_training_err, x=regs, mode='markers+lines', name="Train error"),
               go.Scatter(y=r_validation_err, x=regs, mode='markers+lines', name="Validation error")],
              layout=go.Layout(title="Ridge", yaxis_title="Error",
                               xaxis_title=r"$\lambda$", xaxis_title_font_size=30)).show()

    go.Figure([go.Scatter(y=l_training_err, x=regs, mode='markers+lines', name="Train error"),
               go.Scatter(y=l_validation_err, x=regs, mode='markers+lines', name="Validationerror")],
              layout=go.Layout(title="Lasso", yaxis_title="Error",
                               xaxis_title=r"$\lambda$", xaxis_title_font_size=30)).show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_k_star = regs[np.argmin(r_validation_err)]
    lasso_k_star = regs[np.argmin(l_validation_err)]
    ridgard = RidgeRegression(ridge_k_star).fit(x_train,y_train)
    lasso = Lasso(lasso_k_star).fit(x_train,y_train)
    lin_reg = LinearRegression().fit(x_train,y_train)

    ridge_err = round(mean_square_error(test_y, ridgard.predict(test_x)), 2)
    lass_err = round(mean_square_error(test_y, lasso.predict(test_x)), 2)
    lin_err = lin_reg.loss(test_x,test_y)
    print("**************** Q8 ****************")
    print(f"Ridge error is {ridge_err} with lambda of: {np.round(ridge_k_star,3)}")
    print(f"Lasso error is {lass_err} with lambda of: {np.round(lasso_k_star,3)}")
    print(f"Linear Regression error is {lin_err}")


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree(noise=5)
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(n_samples=1500,noise=10)
    select_regularization_parameter()


