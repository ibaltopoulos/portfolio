import numpy as np
#from sklearn import datasets
#import scipy.stats as ss
#from scipy.optimize import minimize
#import sklearn
from cvxopt import matrix, solvers
import scipy.cluster.hierarchy as sch
import pandas as pd
import random

#port_opt
#portfolioopt
#autofolio

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

def kl_divergence(mu1, cov1, mu2, cov2):
    """
    p1 is true distribution
    """
    x = np.random.multivariate_normal(mu1, cov1, 100000)
    p1 = multivariate_normal_pdf(x, mu1, cov1)
    p2 = multivariate_normal_pdf(x, mu2, cov2)

    return ss.entropy(p1, p2)/100000, np.sum(ss.multivariate_normal.logpdf(x, mu1, cov1))/100000

def elastic_net(x, l1, l2, positive = False):
    alpha = l1 + l2
    l1_ratio = l1 / alpha

    mod = sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive = positive)
    mod.fit(x)
    return mod

def is_none(x):
    return isinstance(x, type(None))

class Convenience(object):
    def __init__(self):
        pass

    def mle_covariance(self):
        """
        Calculates the MLE unbiased covariance matrix of a matrix x
        of shape (n_samples, n_features).
        """
        return np.cov(self.x, rowvar = False)

    def mle_mean(self):
        return self.x.mean(0)

    def get_corr(self):
        return np.corrcoef(self.x, rowvar=False)

class Estimators(Convenience):
    """
    Estimators for covariance and/or means
    """

    def __init__(self, x, mean = 'mle', cov = 'mle'):
        super(Estimators, self).__init__()
        self.x = np.asarray(x)
        self.n_samples = x.shape[0]
        self.n_assets = x.shape[1]
        self.mean = mean
        self.cov = cov

        self.set_mean_and_cov()
        self.corr = self.get_corr()

    def set_mean_and_cov(self):
        self.mean = self.get_mean()
        self.cov = self.get_cov()

    def get_mean(self):
        if self.mean == 'mle':
            return self.mle_mean()
        elif isinstance(self.mean, str):
            quit("Error: Unknown strategy %s for getting means" % self.mean)

    def get_cov(self):
        if self.cov == 'mle':
            return self.mle_covariance()
        elif isinstance(self.cov, str):
            quit("Error: Unknown strategy %s for getting covariance" % self.cov)



    #def naive_bayes(x, basis, functional, unrestricted, h1, h2):
    #    # fit naive bayes model to the mle covariances
    #    pass


    # TODO
    # covar estimators
    # mean estimators


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

    def hrp(self):
        """
        The Hierarchical Risk Parity approach described in http://dx.doi.org/10.2139/ssrn.2708678 

        """
        self.corr = np.asarray([[1,0.7,0.2],[0.7,1,-0.2],[0.2,-0.2,1]])
        self.n_assets = 3

        # distances between correlations
        D = np.sqrt(0.5*(1-self.corr))
        # distance between columns in the distance matrix
        d = np.sqrt(np.sum((D[:,:,None] - D[:,None,:])**2, axis=0))
        # construct the linkage matrix
        links = np.empty((self.n_assets - 1, 2), dtype = int)
        cluster_distances = np.empty(self.n_assets)
        for i in range(self.n_assets-1):
            # get index of the lowest value in the upper triangle
            cluster_distances[i] = np.inf
            links[i] = [-1, -1]
            for j in range(self.n_assets + i):
                if j in links[:i]:
                    continue
                for k in range(j+1, self.n_assets + i):
                    if k in links[:i]:
                        continue

                    if d[j, k] < cluster_distances[i]:
                        cluster_distances[i] = d[j,k]
                        links[i] = [j,k]

            print(links[i], cluster_distances[i])

            # Calculate distances to center
            du = np.minimum(d[links[i,0]], d[links[i,1]])
            d_new = np.zeros((self.n_assets+i+1, self.n_assets+i+1))
            d_new[:-1, :-1] = d
            d_new[-1, :-1] = du
            d_new[:-1, -1] = du
            d = d_new
        quit()
        #def getQuasiDiag(link):
        #    link=link.astype(int)
        #    sortIx=pd.Series([link[-1,0],link[-1,1]])
        #    numItems=link[-1,3] # number of original items
        #    while sortIx.max()>=numItems:
        #        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
        #        df0=sortIx[sortIx>=numItems] # find clusters
        #        i=df0.index;j=df0.values-numItems
        #        sortIx[i]=link[j,0] # item 1
        #        df0=pd.Series(link[j,1],index=i+1)
        #        sortIx=sortIx.append(df0) # item 2
        #        sortIx=sortIx.sort_index() # re-sort
        #        sortIx.index=range(sortIx.shape[0]) # re-index
        #        return sortIx.tolist()

        #def generateData(nObs, size0, size1, sigma1):
        #    np.random.seed(seed=12345);random.seed(12345)
        #    x=np.random.normal(0,1,size=(nObs,size0)) # each row is a variable
        #    cols=[random.randint(0,size0-1) for i in range(size1)]
        #    y=x[:,cols]+np.random.normal(0,sigma1,size=(nObs,len(cols)))
        #    x=np.append(x,y,axis=1)
        #    x=pd.DataFrame(x,columns=range(1,x.shape[1]+1))
        #    return x,cols

        #def correlDist(corr):
        #    dist=((1-corr)/2.)**.5 # distance matrix
        #    return dist


        #nObs,size0,size1,sigma1=10000,5,5,.25
        #x,cols=generateData(nObs,size0,size1,sigma1)
        #cov,corr=x.cov(),x.corr()
        #dist=correlDist(corr)
        #link = sch.linkage(dist, 'single')
        #sortIx=getQuasiDiag(link)
        #print(sortIx)
        #print(corr)
        #sortIx=corr.index[sortIx].tolist() # recover labels
        #df0=corr.loc[sortIx,sortIx] # reorder

        #quit(sortIx)




#np.random.seed(42)
#true_cov, true_means, x = get_random_matrix(10, 4)
x = np.arange(4).reshape((2,2))
p = Portfolio(x, positive = 0, portfolio = 'hrp', min_upper_bound__n = 4)
#print(p.optimal_portfolio)

