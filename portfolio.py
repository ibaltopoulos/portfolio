import numpy as np
from sklearn import datasets
import scipy.stats as ss
#from scipy.optimize import minimize
#import sklearn
import cvxopt
import scipy.cluster.hierarchy as sch
import pandas as pd
import random
import sklearn.covariance
import sklearn.model_selection
import sklearn.mixture
import sklearn.linear_model
import warnings
import pomegranate
import sys
import matplotlib.pyplot as plt
import inspect
import warnings
import sklearn.exceptions
from sklearn.exceptions import ConvergenceWarning

#def get_random_matrix(n_samples, n_features):
#    cov = datasets.make_spd_matrix(n_features)
#    means = (np.random.random(n_features) - 0.5) * 5
#    x = np.random.multivariate_normal(means, cov, n_samples)
#    return cov, means, x

def lineno():
    return inspect.currentframe().f_back.f_lineno

def multivariate_normal_pdf(x, mu, cov):
    return ss.multivariate_normal.logpdf(x, mu, cov, allow_singular=True)

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
        elif is_none(self.mean_estimator):
            return None
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
        elif is_none(self.cov_estimator):
            return None
        else:
            quit("Error: Unknown strategy %s for getting covariance" % self.cov)

    def naive_bayes_mean(self):
        pass


class Portfolio(Estimators):
    """
    For creating portfolios.
    """

    def __init__(self, df = None, x = None, cost = None, classes = None, mean_estimator = 'mle',
            cov_estimator = 'mle', portfolio = 'zero_mean_min_variance', l2 = 0.0,
            positive_constraint = False, upper_mean_bound = 1, n_mixtures = 1, scaling = False):

        self.x = x
        self.cost = cost
        self.classes = classes
        self.positive_constraint = positive_constraint
        self.portfolio = portfolio
        self.upper_mean_bound = upper_mean_bound
        self.l2 = l2

        # preprocess if a pandas dataframe was given
        self._pandas_parser(df)
        # Get mixture weights if n_mixture > 1
        #self._get_mixture_weights(n_mixtures)


        self.n_samples = self.x.shape[0]
        self.n_assets = self.x.shape[1]

        super(Portfolio, self).__init__(mean_estimator = mean_estimator, 
                cov_estimator = cov_estimator)

        # add small number to the covariance matrix diagonal to make sure that the
        # matrix is not singular
        #self.cov += np.identity(self.n_assets)*1e-6

    def _get_mixture_weights(self, n):
        """
        Fit multivariate normals to the energies under the assumption
        that the data is described by n mixtures.
        """

    def _pandas_parser(self, df):
        if is_none(df):
            return
        # just to make sure that stuff is sorted
        # supress warning as this works like intended
        pd.options.mode.chained_assignment = None
        df.sort_values(['functional', 'basis', 'unrestricted', 'reaction'])
        pd.options.mode.chained_assignment = "warn"

        unique_reactions = df.reaction.unique()
        unique_basis = df.basis.unique()
        unique_functionals = df.functional.unique()

        basis_to_id = {key:value for value, key in enumerate(unique_basis)}
        func_to_id = {key:value for value, key in enumerate(unique_functionals)}
        unres_to_id = {True: 1, False: 0}

        energies = []
        times = []
        errors = []
        for idx, reac in enumerate(unique_reactions):
            sub_df = df.loc[df.reaction == reac]
            energies.append(sub_df.energy.tolist())
            errors.append(sub_df.error.tolist())
            times.append(sub_df.time.tolist())
            if idx == 0:
                func = [func_to_id[x] for x in sub_df.functional.tolist()]
                bas = [basis_to_id[x] for x in sub_df.basis.tolist()]
                unres = [unres_to_id[x] for x in sub_df.unrestricted.tolist()]
                classes = np.asarray([func, bas, unres], dtype=int)

        self.x = np.asarray(errors)
        self.raw = np.asarray(energies)
        self.cost = np.asarray(times)
        self.classes = classes

    def fit(self):
        if self.portfolio == 'zero_mean_min_variance':
            self.weights = self.zero_mean_min_variance()
            self.intercept = 0
        elif self.portfolio == 'min_variance_upper_mean_bound':
            self.weights = self.min_variance_upper_mean_bound()
            self.intercept = 0
        elif self.portfolio == 'min_squared_mean':
            self.weights= self.min_squared_mean()
            self.intercept = 0
        elif self.portfolio == 'elastic_net':
            self.weights, self.intercept = self.elastic_net()
        elif self.portfolio == 'constrained_elastic_net':
            self.weights = self.constrained_elastic_net()
            self.intercept = 0
        elif self.portfolio == 'constrained_elastic_net_cv':
            self.weights = self.constrained_elastic_net_cv()
            self.intercept = 0
        else:
            quit("Error: Unknown portfolio method %s" % self.portfolio)

    def zero_mean_min_variance(self):
        """
        Minimize x'Cx, where C is the covariance matrix and x is the portfolio weights.
        The constraints sum(x) = 1 and m'x = 0 is used, with m being the asset means.
        Optionally the constraint x >= 0 is used if self.positive_constraint == False.
        """

        # objective
        P = cvxopt.matrix(self.cov)
        q = cvxopt.matrix(0.0, (self.n_assets,1))

        #### constraints ###

        # optional constraint x >= 0 if positive == False
        if self.positive_constraint:
            G = cvxopt.matrix(-np.identity(self.n_assets))
            h = cvxopt.matrix(0.0, (self.n_assets, 1))
        else:
            G = cvxopt.matrix(0.0, (1, self.n_assets))
            h = cvxopt.matrix(0.0)

        # sum(x) = 1
        A1 = np.ones((1, self.n_assets))
        b1 = np.ones((1,1))
        # mean.T dot x = 0
        A2 = self.mean.reshape(1, self.n_assets)
        b2 = np.zeros((1,1))
        # combine them
        A = cvxopt.matrix(np.concatenate([A1, A2]))
        b = cvxopt.matrix(np.concatenate([b1, b2]))
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.asarray(sol['x']).ravel()

    def min_variance_upper_mean_bound(self):
        """
        Minimize x'Cx, where C is the covariance matrix and x is the portfolio weights.
        The constraints sum(x) = 1 and |m'x| < self.upper_mean_bound is used, with m being the asset means.
        Optionally the constraint x >= 0 is used if self.positive_constraint == False.
        """

        ### objectives ###
        P = cvxopt.matrix(self.cov)
        q = cvxopt.matrix(0.0, (self.n_assets,1))

        #### constraints ###

        # |m'x| < self.upper_mean_bound as well as
        # optional constraint x >= 0 if positive == False
        if self.positive_constraint:
            G = np.empty((self.n_assets + 2, self.n_assets))
            G[:-2, :] = -np.identity(self.n_assets)
            G[-2:, :] = self.mean
            G[-1:, :] = -self.mean
            G = cvxopt.matrix(G)
            h = np.zeros((self.n_assets+2, 1))
            h[-2:] = self.upper_mean_bound
            h = cvxopt.matrix(h)
        else:
            G = np.zeros((2, self.n_assets))
            G[0,:] = self.mean
            G[1,:] = -self.mean
            G = cvxopt.matrix(G)
            h = np.zeros((2,1))
            h[:] = self.upper_mean_bound
            h = cvxopt.matrix(h)


        # sum(x) = 1
        A = cvxopt.matrix(1.0, (1, self.n_assets))
        b = cvxopt.matrix(1.0)

        ### solve ###

        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.asarray(sol['x']).ravel()

    def min_squared_mean(self):
        """
        Minimize x'mm'x + x'Cx, where C is the covariance matrix, m being the asset means and x is the portfolio weights.
        The constraints sum(x) = 1 is used.
        Optionally the constraint x >= 0 is used if self.positive_constraint == False.
        """

        # objective
        P = cvxopt.matrix(self.mean[:, None] * self.mean[None, :] + self.cov)
        q = cvxopt.matrix(0.0, (self.n_assets,1))

        #### constraints ###

        # optional constraint x >= 0 if positive == False
        if self.positive_constraint:
            G = cvxopt.matrix(-np.identity(self.n_assets))
            h = cvxopt.matrix(0.0, (self.n_assets, 1))
        else:
            G = cvxopt.matrix(0.0, (1, self.n_assets))
            h = cvxopt.matrix(0.0)

        # sum(x) = 1
        A = cvxopt.matrix(1.0, (1, self.n_assets))
        b = cvxopt.matrix(1.0)
        # suppress output
        cvxopt.solvers.options['show_progress'] = False

        # solve
        sol = cvxopt.solvers.qp(P, q, G, h, A, b, verbose=False)
        return np.asarray(sol['x']).ravel()

    def elastic_net(self):
        """
        Minimize the squared error of a linear model with l1 and l2 regularization.
        The regularization parameters are determined by 5 fold cross validation.
        """

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        cv_generator = sklearn.model_selection.RepeatedKFold(n_splits = 5, n_repeats = 5)
        model = sklearn.linear_model.ElasticNetCV(
                cv = cv_generator, 
                l1_ratio = [0.0001, 0.001, 0.01, 0.1, .2, 0.5, 0.8, 0.9, 0.99, 1.0], eps = 1e-3, 
                n_alphas = 20, max_iter = 20, positive = self.positive_constraint,
                #l1_ratio = 0,
                #alphas = 10**np.arange(-10,10,0.5),
                n_jobs=1)


        model.fit(self.raw, (self.raw - self.x)[:,0])

        return model.coef_, model.intercept_

    def constrained_elastic_net(self, x = None, l2 = None, init_weights = None):
        """
        Minimize x'(L+EE')x, where L is a constant times the identity matrix (l2 regularization),
        E is the error of the training set and x is the portfolio weights.
        The constraints sum(x) = 1 is used. Optionally the constraint x >= 0 is used if self.positive_constraint == False.
        """

        if is_none(x):
            x = self.x
        if is_none(l2):
            l2 = self.l2

        ### objectives ###
        P = cvxopt.matrix(x.T.dot(x) + l2*np.identity(self.n_assets))
        q = cvxopt.matrix(0.0, (self.n_assets,1))

        #### constraints ###

        # optional constraint x >= 0 if positive == False
        if self.positive_constraint:
            G = cvxopt.matrix(-np.identity(self.n_assets))
            h = cvxopt.matrix(0.0, (self.n_assets, 1))
        else:
            # This doesn't really work and no idea why, but it isn't used
            G = cvxopt.matrix(0.0, (1, self.n_assets))
            h = cvxopt.matrix(0.0)


        # sum(x) = 1
        A = cvxopt.matrix(1.0, (1, self.n_assets))
        b = cvxopt.matrix(1.0)

        # suppress output
        cvxopt.solvers.options['show_progress'] = False

        ### solve ###
        if is_none(init_weights):
            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        else:
            # warmstart
            sol = cvxopt.solvers.qp(P, q, G, h, A, b, initvals = {'x':cvxopt.matrix(init_weights[:,None])})
        return np.asarray(sol['x']).ravel()

    def constrained_elastic_net_cv(self):
        """
        Determine the optimal l2 value for constrained elastic net
        by 5x5 repeated k-fold cross validation
        """

        cv_generator = sklearn.model_selection.RepeatedKFold(n_splits = 5, n_repeats = 3)

        l2 = 10**np.arange(-6, 3, 1.0)

        rme = np.zeros(l2.size)

        weights = None
        all_weights = []
        for (train, test) in cv_generator.split(range(self.n_samples)):
            for i, v in enumerate(l2):
                if is_none(weights):
                    weights = self.constrained_elastic_net(x = self.x[train], l2 = v)
                else:
                    weights = self.constrained_elastic_net(x = self.x[train], l2 = v, init_weights = weights)
                all_weights.append(weights[:])
                rme[i] += sum(np.sum(weights * self.x[test], axis=1)**2)

        best_idx = np.argmin(rme)
        return self.constrained_elastic_net(l2 = l2[best_idx], init_weights = all_weights[best_idx])

    def get_best(self):
        uniq_func = df.functional.unique()
        uniq_basis = df.basis.unique()
        uniq_reac = df.reaction.unique()
        mae = np.empty((uniq_func.size, uniq_basis.size, 2, uniq_reac.size))
        for i, func in enumerate(df.functional.unique()):
            for j, basis in enumerate(df.basis.unique()):
                for k, unres in enumerate([True, False]):
                    mae[i,j,k] = abs(df.loc[(df.functional == func) & (df.basis == basis) & (df.unrestricted == unres)].error.as_matrix())
        mae_ravel = mae.reshape(-1, mae.shape[-1]).T
        naive = []
        for i in range(mae.shape[-1]):
            x = np.concatenate([mae_ravel[:i], mae_ravel[i+1:]])
            #best = np.argmin(np.sum(x, axis=0)) # mae
            best = np.argmin(np.sum(x**2, axis=0)) # rmsd
            #best = np.argmin(np.max(x, axis=0)) # max
            naive.append(mae_ravel[i,best])

        print(np.max(naive), np.mean(naive))


def outer_cv(df, kwargs):

    reactions = df.reaction.unique()

    portfolio_energies = []
    likelihoods = []
    c = 0
    for (train_idx, test_idx) in sklearn.model_selection.LeaveOneOut().split(reactions):
        print(c)
        c += 1
        reac = reactions[test_idx[0]]
        energies = df.loc[df.reaction == reac].energy.as_matrix()
        target = (energies - df.loc[df.reaction == reac].error.as_matrix())[0]
        timings = df.loc[df.reaction == reac].time.as_matrix()

        train_df = df.loc[df.reaction != reac]

        m = Portfolio(df = train_df, **kwargs)
        m.fit()
        #cut = 1e-6
        #portfolio_energy = np.sum(np.clip(m.optimal_portfolio,cut, 1) / sum(np.clip(m.optimal_portfolio,cut, 1)) * energies)
        portfolio_energies.append(sum(m.weights * energies) + m.intercept - target)
        likelihoods.append(multivariate_normal_pdf(energies, m.mean, m.cov))

    portfolio_energies = np.asarray(portfolio_energies)

    ref_df = df.loc[(df.functional == 'M06-2X') & (df.basis == 'qzvp') & (df.unrestricted == True)][["reaction","error"]]
    #print(ref_df)
    ref = ref_df.error.as_matrix()

    #plt.scatter(portfolio_energies, likelihoods)
    #plt.show()




    print(abs(portfolio_energies).max(), np.median(abs(portfolio_energies)), np.mean(abs(portfolio_energies)))
    #fig, ax = plt.subplots()

    #ax.scatter(abs(portfolio_energies), abs(ref))
    #lims = [
    #        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    #        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    #        ]
    #
    ## now plot both limits against eachother
    #ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    #ax.set_aspect('equal')
    #ax.set_xlim(lims)
    #ax.set_ylim(lims)

    #plt.show()

def evaluate_all_methods(df):

    #outer_cv(df, {"positive_constraint": 0, "portfolio": "elastic_net"})
    #outer_cv(df, {"positive_constraint": 1, "portfolio": "elastic_net"})
    outer_cv(df, {"positive_constraint": 1, "portfolio": "constrained_elastic_net"})
    outer_cv(df, {"positive_constraint": 1, "portfolio": "constrained_elastic_net_cv"})





### TODO
# scaling
# timings
# weights > -1 / -2 etc.
# bayes (pomegranate / pymc)
# mixtures
# elastic net
# t-distribution
# classification of error
# support both elastic net and portfolio methods
# cv distribution / means, cov etc.
# predict call

# Tasks
# compare methods using all data points (linear, normal, t, mixture)/(binary scaling, linear scaling)
# select best
# Repeat for maximum different basis sets
# Time vs accuracy
# Classification of error from probability

if __name__ == "__main__":

    if len(sys.argv) == 1:
        "Example usage: python portfolio abde12_reac.pkl"

    df = pd.read_pickle(sys.argv[1])





    #evaluate_all_methods(df)

