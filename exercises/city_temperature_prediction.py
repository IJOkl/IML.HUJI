import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df[df["Temp"] > -70]
    df['DayofYear'] = df['Date'].dt.dayofyear
    df = df.drop("Date", axis=1)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    temp_df = load_data("C:\\Users\\klain\\IML.HUJI\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # part 1: the polynomial degree should be 2, the plot looks like a Parabola
    israel_df = temp_df.loc[temp_df["Country"] == "Israel"]
    plot = px.scatter(data_frame=israel_df, x=israel_df.DayofYear, y=israel_df.Temp, color=israel_df.Year
                      ,title="Temperature change as function of DayofYear in Israel ")
    plot.show()
    # part 2:
    std_months_israel_df = israel_df.groupby(['Month']).agg(np.std).Temp
    plot2 = px.bar(data_frame=std_months_israel_df, x=std_months_israel_df.index, y=std_months_israel_df,
                   title="Standard deviation of temp in each month in israel",
                   labels={'x': "Month", 'y': 'Standard deviation of temp'})
    plot2.show()

    # Question 3 - Exploring differences between countries
    q3_df2 = temp_df.groupby(['Country', 'Month']).Temp.agg(["mean", "std"]).reset_index()
    plot3 = px.line(data_frame=q3_df2, x='Month', y='mean', color='Country', error_y='std',
                    title="Average Temperature in countries by Months")
    plot3.show()

    # Question 4 - Fitting model for different values of `k`
    X = israel_df.DayofYear
    y = israel_df.Temp
    train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion=0.75)
    loss_for_k = []
    for k in range(1, 11):
        poli = PolynomialFitting(k)
        poli.fit(train_X.to_numpy(), train_y.to_numpy())
        k_loss = poli.loss(test_X.to_numpy(), test_y.to_numpy())
        loss_for_k.append(k_loss)
        print(f"The loss for k={k} is: {k_loss}")
    plot4 = px.bar(x=range(1, 11), y=loss_for_k, title="Loss as function of degree",
                   labels={"x": "Degree", "y": "Loss"})
    plot4.show()

    # Question 5 - Evaluating fitted model on different countries
    loss_per_country = []
    q5poli = PolynomialFitting(5)
    q5poli.fit(X, y)
    no_israel_df = temp_df[temp_df.Country != "Israel"]
    countries = no_israel_df.Country.unique()
    for country in countries:
        country_df = no_israel_df[no_israel_df.Country == country]
        c_loss = q5poli.loss(country_df.DayofYear, country_df.Temp)
        loss_per_country.append(c_loss)
    plot5 = px.bar(x=countries, y=loss_per_country, title="Loss as function of Country",
                   labels={"x": "Country name", "y": "Loss"})
    plot5.show()
