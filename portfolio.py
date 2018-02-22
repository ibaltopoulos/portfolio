import numpy as np
from sklearn import datasets
import scipy.stats as ss
#from scipy.optimize import minimize
#import sklearn
from cvxopt import matrix, solvers
import scipy.cluster.hierarchy as sch
import pandas as pd
import random
import sklearn.covariance
import sklearn.model_selection
import warnings
import pomegranate
import sys

def get_random_matrix(n_samples, n_features):
    cov = datasets.make_spd_matrix(n_features)
    means = (np.random.random(n_features) - 0.5) * 5
    x = np.random.multivariate_normal(means, cov, n_samples)
    return cov, means, x

def multivariate_normal_pdf(x, mu, cov):
    return ss.multivariate_normal.pdf(x, mu, cov)

def shannon_entropy(x, mu, cov):
    p = multivariate_normal_pdf(x, mu, cov)
    return sum(p*np.log(p))

def kl_divergence(mu1, cov1, mu2, cov2, n = int(1e6)):
    """
    p1 is true distribution
    """
    x = np.random.multivariate_normal(mu1, cov1, n)
    p1 = multivariate_normal_pdf(x, mu1, cov1)
    p2 = multivariate_normal_pdf(x, mu2, cov2)

    return ss.entropy(p1, p2)

def is_none(x):
    return isinstance(x, type(None))

class NaiveBayesRegression(object):
    """
    Naive bayes model, where the number of hidden nodes acts as regularization for the data.

    """

    def __init__(self):
        pass

    # TODO fix
    def fit(self, x, distributions, n_nodes = 3):
        """
        Fit the hidden bayes model.
        x: array of size (n_samples, n_features)
        distributions: array of size (n_features, ) that indicates which distribution the 
            features follows. Can be any singlevariate distribution from pomegranate

        """

        init_dist = []
        for i, dist in enumerate(distributions):
            if isinstance(dist, DiscreteDistribution):
                pass


        product_distribution = pomegranate.distributions.IndependentComponentsDistribution()

        self.model = pomegranate.GeneralMixtureModel.from_samples(distributions, n_components = n_nodes, X = x)



class Estimators(object):
    """
    Estimators for covariance and/or means
    """

    def __init__(self, x, mean = 'mle', cov = 'mle'):
        self.x = np.asarray(x)
        self.n_samples = x.shape[0]
        self.n_assets = x.shape[1]
        self.mean = mean
        self.cov = cov

        self._set_mean_and_cov()
        self.corr = self.get_mle_corr()

    def get_mle_covariance(self, x = None, ddof = 1):
        """
        Calculates the MLE unbiased covariance matrix of a matrix x
        of shape (n_samples, n_features).
        """
        if is_none(x):
            x = self.x

        # 1e-7 to avoid singularities
        return np.cov(x, ddof = 1, rowvar = False) + 1e-7 * np.identity(x.shape[1])

    def get_oas_covariance(self, x = None):
        """
        Calculates the OAS (Oracle Approximating Shrinkage Estimator) covariance.
        """
        if is_none(x):
            x = self.x

        return sklearn.covariance.oas(self.x)[0]

    def get_lw_covariance(self, x = None):
        """
        Calculates the shrunk Ledoit-Wolf covariance matrix.
        """
        if is_none(x):
            x = self.x

        return sklearn.covariance.ledoit_wolf(self.x)[0]

    def get_gl_covariance(self, x = None, l1 = 0.1):
        """
        Calculates the GraphLasso variance, where the off diagonal elements have l1-regularization
        """

        if is_none(x):
            x = self.x

        empirical_cov = self.get_mle_covariance(x = x)
        return sklearn.covariance.graph_lasso(emp_cov = empirical_cov, alpha=l1, max_iter=300)[0]

    def get_gl_covariance(self, x = None, alpha = 0.1):
        """
        Calculates the GraphLasso variance, where the off diagonal elements have l1-regularization
        """

        if is_none(x):
            x = self.x

        empirical_cov = self.get_mle_covariance(x = x)
        return sklearn.covariance.graph_lasso(emp_cov = empirical_cov, alpha=alpha, max_iter=100)[0]

    def get_gl_covariance_cv(self, x = None):
        """
        Leave-one-out cross validation of the covariance matrix by a Graphical Lasso.
        """

        return self.leave_one_out_cv(self.get_gl_covariance, x, lb = -5, ub = 5)

    def get_shrunk_covariance(self, x = None, alpha = 0.1):
        """
        Calculates the shrunk covariance
        """

        if is_none(x):
            x = self.x

        empirical_cov = self.get_mle_covariance(x = x)
        return sklearn.covariance.shrunk_covariance(emp_cov = empirical_cov, shrinkage=alpha)

    def get_shrunk_covariance_cv(self, x = None):
        """
        Leave-one-out cross validation of the covariance matrix by a Graphical Lasso.
        """

        return self.leave_one_out_cv(self.get_shrunk_covariance, x, lb = 0, ub = 1, logscale = False)

    def get_mincovdet_covariance(self, x = None, alpha = None):
        """
        Calculates the shrunk covariance
        """

        if is_none(x):
            x = self.x

        mod = sklearn.covariance.MinCovDet(support_fraction = alpha)
        mod.fit(x)
        return mod.covariance_

    def leave_one_out_cv(self, est, x = None, lb = 1e-5, ub = 1e5, logscale = True):

        if is_none(x):
            x = self.x

        mu = self.get_mle_mean(x)

        best_ll  = np.inf
        if logscale:
            values = np.logspace(lb, ub, num=100)
        else:
            values = np.linspace(lb, ub, num=100)
        for alpha in values:
            ll = 0
            for (train_idx, test_idx) in sklearn.model_selection.LeaveOneOut().split(x):
                train = x[train_idx]
                test = x[test_idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        train_cov = est(x = train, alpha=alpha)
                    except FloatingPointError:
                        train_cov = np.identity(self.n_assets)
                    ll += -np.log(multivariate_normal_pdf(test, mu, train_cov))
            if ll < best_ll:
                best_ll = ll
                best_alpha = alpha
        return est(x = x, alpha = best_alpha)

    def get_mle_mean(self, x = None):
        if is_none(x):
            x = self.x

        return x.mean(0)

    def get_mle_corr(self, x = None):
        if is_none(x):
            x = self.x

        return np.corrcoef(x, rowvar=False)

    def _set_mean_and_cov(self):
        self.mean = self._get_mean()
        self.cov = self._get_cov()

    def _get_mean(self):
        if self.mean == 'mle':
            return self.get_mle_mean()
        elif isinstance(self.mean, str):
            quit("Error: Unknown strategy %s for getting means" % self.mean)

    def _get_cov(self):
        if self.cov == 'mle':
            return self.get_mle_covariance()
        elif self.cov == 'oas':
            return self.get_oas_covariance()
        elif self.cov == 'gl':
            return self.get_gl_covariance_cv()
        elif self.cov == 'lw':
            return self.get_lw_covariance()
        elif isinstance(self.cov, str):
            quit("Error: Unknown strategy %s for getting covariance" % self.cov)

    def naive_bayes_mean(self):
        pass

class Portfolio(Estimators):
    """
    For creating portfolios
    """

    def __init__(self, x, mean = 'mle', cov = 'mle', portfolio = 'zero_mean_min_variance', 
            positive = False, min_upper_bound__n = 1):
        super(Portfolio, self).__init__(x = x, mean = mean, cov = cov)

        self.positive = positive
        self.portfolio = portfolio
        self.min_upper_bound__n = min_upper_bound__n

        self.optimal_portfolio = self.get_optimal_portfolio()

    def get_optimal_portfolio(self):
        if self.portfolio == 'zero_mean_min_variance':
            return self.zero_mean_min_variance()
        elif self.portfolio == 'min_upper_bound':
            return self.min_upper_bound()
        elif self.portfolio == 'hrp':
            return self.hrp()
        quit("Error: Unknown portfolio method %s" % self.portfolio)


    def zero_mean_min_variance(self):
        """
        Minimize x'Cx, where C is the covariance matrix and x is the portfolio weights.
        The constraints sum(x) = 1 and m'x = 0 is used, with m being the asset means.
        Optionally the constraint x >= 0 is used if positive == False.
        """

        # objective
        P = matrix(self.cov)
        q = matrix(0.0, (self.n_assets,1))

        #### constraints ###

        # optional constraint x >= 0 if positive == False
        if self.positive:
            G = matrix(-np.identity(self.n_assets))
            h = matrix(0.0, (self.n_assets, 1))
        else:
            G = matrix(0.0, (1, self.n_assets))
            h = matrix(0.0)

        # sum(x) = 1
        A1 = np.ones((1, self.n_assets))
        b1 = np.ones((1,1))
        # mean.T dot x = 0
        A2 = self.mean.reshape(1, self.n_assets)
        b2 = np.zeros((1,1))
        # combine them
        A = matrix(np.concatenate([A1, A2]))
        b = matrix(np.concatenate([b1, b2]))
        sol = solvers.qp(P, q, G, h, A, b)
        return np.asarray(sol['x']).ravel()

    def opt(mu, cov):
        """
        minimize    (x -mu )' Q (x - mu)
        subject to  x > 0
        where Q is the precision matrix cov^(-1)

        CVXOPT minimizes    x'Px + q'x 
        subject to          Gx <= h
                            Ax == b

        The objective can be rewritten as x'Qx - 2 mu' Q x
        since mu' Q mu is a constant.

        So gathering up the terms we have that P = Q, 
        q' = -2 mu' Q, G = -1, A = 0, b = 0 and h = 0

        mu: means of shape n_samples
        cov: covariance matrix of shape (n_features, n_features)

        """
        from cvxopt import matrix, solvers

        n_features = cov.shape[0]
        Q = np.linalg.inv(cov)

        # objective
        P = matrix(precision)
        q = matrix(-2*mu[None, :].dot(Q))

        # constraints

        G = matrix(-np.identity(n_features))
        A = matrix(0.0, (n_features, n_features))
        b = matrix(0.0, (n_features, 1))
        h = matrix(0.0, (n_features, 1))

        sol = solvers.qp(P, q, G, h, A, b)
        return np.asarray(sol['x']).ravel()

    def min_upper_bound(self):
        """
        Minimize x'(nC+mm')x, where C is the covariance matrix, x is the portfolio weights,
        m is the asset means and n is a constant.
        The 1d correspondence to n=1 is to minimize the 68 percentile and for n=4 the 95 percentile.
        The constraints sum(x) = 1 is used.
        Optionally the constraint x >= 0 is used if positive == False.
        """
        n = self.min_upper_bound__n
        # objective
        m2 = np.dot(self.mean[:,None], self.mean[None,:])
        # TODO remove later
        assert(m2.size == self.n_assets**2)

        P = matrix(n * self.cov + m2)
        q = matrix(0.0, (self.n_assets,1))

        #### constraints ###

        # optional constraint x >= 0 if positive == False
        if self.positive:
            G = matrix(-np.identity(self.n_assets))
            h = matrix(0.0, (self.n_assets, 1))
        else:
            G = matrix(0.0, (1, self.n_assets))
            h = matrix(0.0)

        # sum(x) = 1
        A = matrix(np.ones((1, self.n_assets)))
        b = matrix(np.ones((1,1)))
        sol = solvers.qp(P, q, G, h, A, b)
        return np.asarray(sol['x']).ravel()

if __name__ == "__main__":

    if len(sys.argv) == 1:
        "Example usage: python portfolio abde12_reac.pkl"

    df = pd.read_pickle(sys.argv[1])


    e = Estimators(x)
    #cov = e.get_gl_covariance_cv()
    #print(kl_divergence(true_mu, true_cov, true_mu, cov))
    #cov = e.get_shrunk_covariance_cv()
    #print(kl_divergence(true_mu, true_cov, true_mu, cov))
    cov = e.get_oas_covariance()
    print(kl_divergence(true_mu, true_cov, true_mu, cov))
    cov = e.get_mincovdet_covariance()
    print(kl_divergence(true_mu, true_cov, true_mu, cov))
    #p = Portfolio(x, positive = 0, cov='mle', portfolio = 'hrp', min_upper_bound__n = 4)
    #print(p.optimal_portfolio)

