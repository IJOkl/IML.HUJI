from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    print("********* Q1 *********")

    # Question 1 - Draw samples and print fitted model
    uni = UnivariateGaussian()
    X = np.random.normal(10, 1, 1000)
    uni.fit(X)
    print((uni.mu_, uni.var_))

    # Question 2 - Empirically showing sample mean is consistent
    print("********* Q2 *********")

    ms = np.linspace(10, 1000, num=100).astype(int)
    distances = []

    for m in ms:
        distances.append(np.abs(uni.mu_ - np.mean(X[0:m])))

    go.Figure([go.Scatter(x=ms, y=distances, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Distance between the estimated and true value of the expectation}$",
                               xaxis_title=r"$\text{Number of samples}$",
                               yaxis_title="r$|\hat\mu - \mu|$",
                               height=350)).show()
    print("********* Q3 *********")
    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = uni.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdfs, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Sample values and corresponding PDF's}$",
                               xaxis_title="$\\text{ Sample values }$",
                               yaxis_title="$\\text{PDF of sample value}$",
                               height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    print("********* Q4 *********")
    mult_mu = np.array([0, 0, 4, 0])
    mult_cov = np.array([[1, 0.2, 0, 0.5],
                         [0.2, 2, 0, 0],
                         [0, 0, 1, 0],
                         [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mult_mu, mult_cov, size=(1000,))
    multi = MultivariateGaussian()
    multi.fit(X)
    print("Estimated Expectation: ", multi.mu_)
    print("Covariance Matrix: ")
    print(multi.cov_)

    # Question 5 - Likelihood evaluation
    f1_space = np.linspace(-10, 10, 200)
    f3_space = np.linspace(-10, 10, 200)
    log_like = []
    for i in range(len(f1_space)):
        log_like.append([])
        for j in range(len(f3_space)):
            log_like[i].append(MultivariateGaussian.log_likelihood(np.array([f1_space[i], 0, f3_space[j], 0]),mult_cov,X))


    heatmap = go.Figure(data = go.Heatmap(x = f3_space,y =f1_space,z = log_like))
    heatmap.update_layout(showlegend = True,autosize = True,title = "Q5 Heatmap",
                          xaxis_title = "$\\text{ f3 Values }$",yaxis_title = "$\\text{ f1 Values }$")
    heatmap.show()

    # Question 6 - Maximum likelihood
    maxi = np.max(log_like)
    arg_max = np.argmax(log_like)
    i = arg_max // 200
    j = arg_max % 200
    f1_max = f1_space[i]
    f3_max = f3_space[j]
    print("Max value is: ",maxi)
    print("f1 and f3 arg max:")
    print(f1_max,f3_max)

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
