from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni = UnivariateGaussian()
    X = np.random.normal(10,1,1000)
    uni.fit(X)
    print((uni.mu_,uni.var_))

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10,1000,num=100).astype(int)
    distances = []

    for m in ms:
        distances.append(np.abs(uni.mu_ - np.mean(X[0:m])))

    go.Figure([go.Scatter(x=ms, y=distances, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Distance between the estimated and true value of the expectation}$",
                               xaxis_title=r"$\text{Number of samples}$",
                               yaxis_title="r$|\hat\mu - \mu|$",
                               height=350)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = uni.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdfs, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Sample values and corresponding PDF's}$",
                               xaxis_title="$\\text{ Sample values }$",
                               yaxis_title="$\\text{PDF of sample value}$",
                               height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
