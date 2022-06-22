import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from utils import custom
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error
import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def call_back(model, **kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])
        return

    return call_back, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    min_g = np.inf
    best_eta = np.inf
    for mod, num in [(L1, 1), (L2, 2)]:

        for step in etas:
            li = mod(init.copy())
            c = get_gd_state_recorder_callback()
            g = GradientDescent(learning_rate=FixedLR(step), callback=c[0], out_type="best")
            g.fit(li, X=None, y=None)
            fig1 = plot_descent_path(mod, np.array(c[2]), title=f"L{num} mod with eta of {step}")
            fig1.show()
            fig3 = go.Figure([go.Scatter(x=list(range(len(c[1]))), y=c[1], mode="markers", marker_color="blue")],
                             layout=go.Layout(title=f"convergence of L{str(num)}, eta: {step}",
                                              xaxis_title="t", yaxis_title="convergence"))
            fig3.show()
            if (min_g > c[1][-1]):
                best_eta = step
                min_g = c[1][-1]
        print(f"min loss for l{num} is {str(min_g)} with eta  {str(best_eta)}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    cr, d = [], []
    for gamma in gammas:
        c = get_gd_state_recorder_callback()
        g = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=gamma), out_type="best", callback=c[0])
        l = L1(init.copy())
        g.fit(l, X=None, y=None)
        cr.append(c[1])
        if gamma == .95:
            d = c[2]

    # Plot algorithm's convergence for the different values of gamma
    plt.plot(list(range(len(cr[0]))), cr[0])
    plt.plot(list(range(len(cr[1]))), cr[1])
    plt.plot(list(range(len(cr[2]))), cr[2])
    plt.plot(list(range(len(cr[3]))), cr[3])
    plt.title("convergence rate and decay rates"), plt.xlabel("t"), plt.ylabel("norm")
    plt.legend(["gamma = 0.9", "gamma = 0.95", "gamma = 0.99", "gamma = 1"])
    plt.grid()
    plt.show()

    print(f"exponentially decay - l1 lowest norm: {np.min([np.min(cr[i]) for i in range(4)])}")
    # Plot descent path for gamma=0.95
    l2 = L2(init.copy())
    c2 = get_gd_state_recorder_callback()
    g2 = GradientDescent(learning_rate=ExponentialLR(eta, .95), callback=c2[0], out_type="best")
    g2.fit(l2, X=None, y=None)

    plot_descent_path(L1, np.array(d), title="descent path l1 model with gamma = 0.95 ").show()
    plot_descent_path(L2, np.array(c2[2]), title="descent path l2 model with gamma = 0.95 ").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train,y_train,X_test,y_test = np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test)
    c = [custom[0],custom[-1]]


    # Plotting convergence rate of logistic regression over SA heart disease data
    mod = LogisticRegression(include_intercept=True
                             ,solver=GradientDescent(learning_rate=FixedLR(1e-4)
                                                     ,max_iter=20000))
    mod.fit(X_train,y_train)
    prob = mod.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, prob)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()
    alpha_idx = np.argmax(tpr-fpr)
    best_alpha = round(thresholds[alpha_idx],2)
    print("**********************************")
    print(f"Best alpha for logistic is {best_alpha} in idx {alpha_idx}")
    print(f"model has a test error of {mod._loss(X_test,y_test)}")
    print("**********************************")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for model in ["l1","l2"]:
        all_err = []
        for lamda in lambdas:
            est = LogisticRegression(include_intercept=True,penalty=model,
                                     solver=GradientDescent(FixedLR(1e-4),max_iter=20000),lam = lamda)
            train_err,val_err = cross_validate(est,X_train,y_train,misclassification_error)
            all_err.append(val_err)
        best_l = lambdas[np.argmin(all_err)]
        m = LogisticRegression(include_intercept=True,penalty=model,
                                   solver=GradientDescent(learning_rate=FixedLR(1e-4),max_iter=20000),lam=best_l)
        m.fit(X_train,y_train)
        test_err = m._loss(X_test,y_test)
        print("************************************************")
        print(f"test error for model {model} is {str(test_err) } with best lambda of {best_l}")
        print("************************************************")





if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    # fit_logistic_regression()
