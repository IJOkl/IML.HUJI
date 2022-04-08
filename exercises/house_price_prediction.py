from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    df['zipcode'] = df['zipcode'].astype(int)
    df = df.drop(df[(df.price <= 0) |
                    (df.bedrooms <= 0) |
                    (df.bathrooms <= 0) |
                    (df.sqft_living <= 0) |
                    (df.floors <= 0) |
                    (df.sqft_above <= 0) |
                    (df.yr_built <= 0)].index)
    df["year_group"] = (df["yr_built"] / 25).astype(int)
    year_group_dummies = pd.get_dummies(df.year_group)
    zip_dummies = pd.get_dummies(df.zipcode)
    df = df.join(zip_dummies)
    df = df.join(year_group_dummies)
    df = df.drop(columns=['date', 'id', 'zipcode', 'long', 'lat', 'yr_built'])
    y = df.price
    X = df.drop(['price'], axis=1)
    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_std = np.std(y)
    for feature, col_data in X.items():
        plt.figure()
        pearson = __pearson(col_data, y, y_std)
        plt.title(f"{feature} vs Price\n"
                  f" Correlation: {pearson}")
        plt.xlabel(feature)
        plt.ylabel('price')
        plt.scatter(col_data, y, color='hotpink')
        # plt.show()
        plt.savefig(fname=f"{output_path}/{feature}.png")


def __pearson(col_data, y, y_std):
    col_std = np.std(col_data)
    return col_data.cov(y) / (col_std * y_std)


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data('C:\\Users\\klain\\IML.HUJI\\datasets\\house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, )

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion=0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    reginald = LinearRegression()

    mean_loss = np.zeros(91)
    loss_std = np.zeros(91)
    for p in range(10, 101):
        p_loss = []
        for i in range(0, 10):
            p_train_X = train_X.sample(frac=p / 100)
            p_train_y = train_y[p_train_X.index]
            reginald.fit(p_train_X.to_numpy(), p_train_y)
            p_loss.append(reginald.loss(test_X.to_numpy(), test_y.to_numpy()))
        mean_loss[p - 10] = np.mean(p_loss)
        loss_std[p - 10] = np.std(p_loss)
    p_s = np.arange(0.1, 1, 0.01)
    fig = go.Figure([go.Scatter(x=p_s, y=mean_loss - 2 * loss_std, fill=None, mode="lines",
                                line=dict(color="lightblue"), showlegend=False),
                     go.Scatter(x=p_s, y=mean_loss + 2 * loss_std, fill='tonexty', mode="lines",
                                line=dict(color="lightblue"), showlegend=False),
                     go.Scatter(x=p_s, y=mean_loss, mode="markers+lines", marker=dict(color="blue", size=1),
                                showlegend=False)],
                    layout=go.Layout(title=f"Mean loss as a function of sample percentage",
                                     height=500))

    fig.show()
