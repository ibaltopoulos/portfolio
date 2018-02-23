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
import sklearn.mixture
import warnings
import pomegranate
import sys
import matplotlib.pyplot as plt

#def get_random_matrix(n_samples, n_features):
#    cov = datasets.make_spd_matrix(n_features)
#    means = (np.random.random(n_features) - 0.5) * 5
#    x = np.random.multivariate_normal(means, cov, n_samples)
#    return cov, means, x

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

    def __init__(self, mean_estimator = 'mle', cov_estimator = 'mle'):
        self.mean_estimator = mean_estimator
        self.cov_estimator = cov_estimator
        self._set_mean_and_cov()
        #self.corr = self.get_mle_corr()


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
        if self.mean_estimator == 'mle':
            return self.get_mle_mean()
        else:
            quit("Error: Unknown strategy %s for getting means" % self.mean_estimator)

    def _get_cov(self):
        if self.cov_estimator == 'mle':
            return self.get_mle_covariance()
        elif self.cov_estimator == 'oas':
            return self.get_oas_covariance()
        elif self.cov_estimator == 'gl':
            return self.get_gl_covariance_cv()
        elif self.cov_estimator == 'lw':
            return self.get_lw_covariance()
        elif isinstance(self.cov, str):
            quit("Error: Unknown strategy %s for getting covariance" % self.cov)

    def naive_bayes_mean(self):
        pass


class Portfolio(Estimators):
    """
    For creating portfolios.
    """

    def __init__(self, df = None, x = None, cost = None, classes = None, mean_estimator = 'mle',
            cov_estimator = 'mle', portfolio = 'zero_mean_min_variance',
            positive_constraint = False, upper_mean_bound = 1, n_mixtures = 1, scaling = False):

        self.x = x
        self.cost = cost
        self.classes = classes
        self.positive_constraint = positive_constraint
        self.portfolio = portfolio
        self.upper_mean_bound = upper_mean_bound

        # preprocess if a pandas dataframe was given
        self._pandas_parser(df)
        # Get mixture weights if n_mixture > 1
        #self._get_mixture_weights(n_mixtures)


        self.n_samples = self.x.shape[0]
        self.n_assets = self.x.shape[1]

        super(Portfolio, self).__init__(mean_estimator = mean_estimator, 
                cov_estimator = cov_estimator)

        self.optimal_portfolio = self.get_optimal_portfolio()

    def _get_mixture_weights(self, n):
        """
        Fit multivariate normals to the energies under the assumption
        that the data is described by n mixtures.
        """


    def _pandas_parser(self, df):
        if is_none(df):
            return
        unique_reactions = df.reaction.unique()
        unique_basis = df.basis.unique()
        unique_functionals = df.functional.unique()

        basis_to_id = {key:value for value, key in enumerate(unique_basis)}
        func_to_id = {key:value for value, key in enumerate(unique_functionals)}
        unres_to_id = {True: 1, False: 0}

        energies = []
        times = []
        for idx, reac in enumerate(unique_reactions):
            sub_df = df.loc[df.reaction == reac]
            energies.append(sub_df.energy.tolist())
            times.append(sub_df.time.tolist())
            if idx == 0:
                func = [func_to_id[x] for x in sub_df.functional.tolist()]
                bas = [basis_to_id[x] for x in sub_df.basis.tolist()]
                unres = [unres_to_id[x] for x in sub_df.unrestricted.tolist()]
                classes = np.asarray([func, bas, unres], dtype=int)

        self.x = np.asarray(energies)
        self.cost = np.asarray(times)
        self.classes = classes


    def get_optimal_portfolio(self):
        if self.portfolio == 'zero_mean_min_variance':
            return self.zero_mean_min_variance()
        elif self.portfolio == 'min_variance_upper_mean_bound':
            return self.min_variance_upper_mean_bound()
        elif self.portfolio == 'min_squared_mean':
            return self.min_squared_mean()
        quit("Error: Unknown portfolio method %s" % self.portfolio)

    def zero_mean_min_variance(self):
        """
        Minimize x'Cx, where C is the covariance matrix and x is the portfolio weights.
        The constraints sum(x) = 1 and m'x = 0 is used, with m being the asset means.
        Optionally the constraint x >= 0 is used if self.positive_constraint == False.
        """

        # objective
        P = matrix(self.cov)
        q = matrix(0.0, (self.n_assets,1))

        #### constraints ###

        # optional constraint x >= 0 if positive == False
        if self.positive_constraint:
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

    def min_variance_upper_mean_bound(self):
        """
        Minimize x'Cx, where C is the covariance matrix and x is the portfolio weights.
        The constraints sum(x) = 1 and |m'x| < self.upper_mean_bound is used, with m being the asset means.
        Optionally the constraint x >= 0 is used if self.positive_constraint == False.
        """

        ### objectives ###
        P = matrix(self.cov)
        q = matrix(0.0, (self.n_assets,1))

        #### constraints ###

        # |m'x| < self.upper_mean_bound as well as
        # optional constraint x >= 0 if positive == False
        if self.positive_constraint:
            G = np.empty((self.n_assets + 2, self.n_assets))
            G[:-2, :] = -np.identity(self.n_assets)
            G[-2:, :] = self.mean
            G[-1:, :] = -self.mean
            G = matrix(G)
            h = np.zeros((self.n_assets+2, 1))
            h[-2:] = self.upper_mean_bound
            h = matrix(h)
        else:
            G = np.zeros((2, self.n_assets))
            G[0,:] = self.mean
            G[1,:] = -self.mean
            G = matrix(G)
            h = np.zeros((2,1))
            h[:] = self.upper_mean_bound
            h = matrix(h)


        # sum(x) = 1
        A = matrix(1.0, (1, self.n_assets))
        b = matrix(1.0)

        ### solve ###

        sol = solvers.qp(P, q, G, h, A, b)
        return np.asarray(sol['x']).ravel()

    def min_squared_mean(self):
        """
        Minimize x'mm'x + x'Cx, where C is the covariance matrix, m being the asset means and x is the portfolio weights.
        The constraints sum(x) = 1 is used.
        Optionally the constraint x >= 0 is used if self.positive_constraint == False.
        """

        # objective
        P = matrix(self.mean[:, None] * self.mean[None, :] + self.cov)
        q = matrix(0.0, (self.n_assets,1))

        #### constraints ###

        # optional constraint x >= 0 if positive == False
        if self.positive_constraint:
            G = matrix(-np.identity(self.n_assets))
            h = matrix(0.0, (self.n_assets, 1))
        else:
            G = matrix(0.0, (1, self.n_assets))
            h = matrix(0.0)

        # sum(x) = 1
        A = matrix(1.0, (1, self.n_assets))
        b = matrix(1.0)
        sol = solvers.qp(P, q, G, h, A, b)
        return np.asarray(sol['x']).ravel()

def outer_cv(df):

    reactions = df.reaction.unique()

    portfolio_energies = []
    for (train_idx, test_idx) in sklearn.model_selection.LeaveOneOut().split(reactions):
        reac = reactions[test_idx[0]]
        energies = df.loc[df.reaction == reac].energy.as_matrix()
        timings = df.loc[df.reaction == reac].time.as_matrix()

        train_df = df.loc[df.reaction != reac]

        #m = Portfolio(df = train_df, positive_constraint = 1, portfolio = 'zero_mean_min_variance', upper_mean_bound = 0)
        #m = Portfolio(df = train_df, positive_constraint = 1, portfolio = 'min_variance_upper_mean_bound', upper_mean_bound = 1)
        m = Portfolio(df = train_df, positive_constraint = 1, portfolio = 'min_squared_mean', upper_mean_bound = 0)
        portfolio_energy = np.sum(m.optimal_portfolio * energies)
        portfolio_energies.append(portfolio_energy)

    portfolio_energies = np.asarray(portfolio_energies)

    pbe0 = df.loc[(df.functional == 'M06-2X') & (df.basis == 'qzvp') & (df.unrestricted == True)].energy.as_matrix()

    fig, ax = plt.subplots()
    print(abs(portfolio_energies).max(), np.median(abs(portfolio_energies)), np.mean(abs(portfolio_energies)))
    print(abs(pbe0).max(), np.median(abs(pbe0)), np.mean(abs(pbe0)))
    #ax.scatter(abs(portfolio_energies), abs(pbe0))

    #lims = [
    #np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    #np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    #]
    #
    ## now plot both limits against eachother
    #ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    #ax.set_aspect('equal')
    #ax.set_xlim(lims)
    #ax.set_ylim(lims)

    #plt.show()



### TODO
# bayes (pomegranate / pymc)
# mixtures
# scaling
# outer cv

if __name__ == "__main__":

    if len(sys.argv) == 1:
        "Example usage: python portfolio abde12_reac.pkl"

    df = pd.read_pickle(sys.argv[1])
    outer_cv(df)
    #m = Portfolio(df = df, positive_constraint = 1)
    #print(m.optimal_portfolio[m.optimal_portfolio > 1e-2])


