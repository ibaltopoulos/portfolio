"""
Generate data for paper

"""

import os, sys
import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
import pickle
import matplotlib.pyplot as plt
import copy

# Add the module to source path
dirname = os.path.dirname(os.path.realpath(__file__))
source_path = dirname + "/portfolio"
sys.path.append(source_path)

from portfolio.model import SingleMethod, NN, LinearModel

def calc_mean_and_var(x):
    vars_ = np.var(x, ddof = 1, axis = 1)
    means = np.mean(x, axis = 1)
    alphas = vars_ / sum(vars_)
    mean = sum(means * alphas)

def run_SingleMethod(x,y, cost = None, names = None, classes = None):
    if os.path.isfile("pickles/single_method_results.pkl"):
        print("single method results already generated")
        return


    m = SingleMethod()
    cv_params = {'loss': ('mae', 'rmsd', 'max')}
    # Separate different classes and methods
    unique_classes = np.unique(classes)
    unique_costs = np.unique(cost)

    d = {}

    for cl in unique_classes:
        params = []
        estimators = []
        errors = []
        cv_portfolios = []
        portfolios = []
        portfolio_weights = []
        portfolio_names = []
        maes = []
        rmsds = []
        maxerrs = []
        sample_maes = []
        sample_rmsds = []
        laplace_mnll = []
        gaussian_mnll = []
        for co in unique_costs:
            class_idx = np.where(classes == cl)[0]
            cost_idx = np.where(cost <= co)[0]
            # Get best hyperparams from cv
            error, portfolio, best_params, cv_portfolio = \
                    outer_cv(x[np.ix_(class_idx, cost_idx)], y[class_idx], m, cv_params)
            # Train model with best hyperparams on full data
            m.set_params(**best_params)
            m.fit(x[np.ix_(class_idx, cost_idx)], y[class_idx])
            # Get human readable details on the selected portfolio
            portfolio_weight, portfolio_name = get_portfolio_details(portfolio, names)
            # Get statistics
            mae, rmsd, maxerr, sample_mae, sample_rmsd, nll_laplace, nll_gaussian = get_error_statistics(error)

            # Store various attributes for the given cost
            params.append(best_params)
            estimators.append(copy.copy(m))
            errors.append(error)
            cv_portfolios.append(cv_portfolio)
            portfolios.append(portfolio)
            portfolio_weights.append(portfolio_weight)
            portfolio_names.append(portfolio_name)
            maes.append(mae)
            rmsds.append(rmsd)
            maxerrs.append(maxerr)
            sample_maes.append(sample_mae)
            sample_rmsds.append(sample_rmsd)
            laplace_mnll.append(nll_laplace)
            gaussian_mnll.append(nll_gaussian)

        # Store various attributes for the given class
        d[cl] = {'errors': errors,
                 'cv_portfolios': cv_portfolios,
                 'portfolios': portfolios,
                 'params': params,
                 'estimators': estimators,
                 'cost': unique_costs,
                 'portfolio_weights': portfolio_weights,
                 'portfolio_names': portfolio_names,
                 'maes': maes,
                 'rmsds': rmsds,
                 'maxerrs': maxerrs,
                 'sample_maes': sample_maes,
                 'sample_rmsds': sample_rmsds,
                 'laplace_mnll': laplace_mnll,
                 'gaussian_mnll': gaussian_mnll
                 }
    
    # Dump the results in a pickle
    with open("pickles/single_method_results.pkl", 'wb') as f:
        pickle.dump(d, f, -1)

def get_error_statistics(errors):

    mae = np.mean(abs(errors))
    rmsd = np.sqrt(np.mean(errors**2))
    maxerr = np.max(abs(errors))

    sample_mae = np.zeros(errors.size)
    sample_rmsd = np.zeros(errors.size)
    for idx, j in sklearn.model_selection.LeaveOneOut().split(errors):
        sample_mae[j] = np.mean(abs(errors[idx]))
        sample_rmsd[j] = np.sqrt(np.mean(errors[idx]**2))

    nll_laplace = np.mean(np.log(2*sample_mae) + abs(errors) / sample_mae)
    nll_gaussian = 0.5 * np.log(2 * np.pi) + np.mean(np.log(sample_rmsd) + errors**2 / sample_rmsd**2)

    return  mae, rmsd, maxerr, sample_mae, sample_rmsd, nll_laplace, nll_gaussian


def run_LinearModel(x,y, cost = None, names = None, classes = None):
    if os.path.isfile("pickles/linear_method_results.pkl"):
        print("Linear method results already generated")
        return


    m = NN(tensorboard_dir = '', learning_rate = 1e-1, iterations = 20000, 
            l2_reg = 1e-6, cost_reg = 0, cost = cost) # cost_reg ~ (1e-6, 1e5) good start

    cost_reg = [0] + [10**i for i in range(-6,5)]
    m.fit(x,y)
    idx = np.argsort(m.portfolio)[-5:]
    print(m.portfolio[idx], sum(m.portfolio > 1e-6))
    quit()
    cv_params = {'loss': ('mae', 'rmsd', 'max')}
    # Separate different classes and methods
    unique_classes = np.unique(classes)
    unique_costs = np.unique(cost)

    d = {}

    for cl in unique_classes:
        params = []
        estimators = []
        errors = []
        cv_portfolios = []
        portfolios = []
        portfolio_weights = []
        portfolio_names = []
        maes = []
        rmsds = []
        maxerrs = []
        sample_maes = []
        sample_rmsds = []
        laplace_mnll = []
        gaussian_mnll = []
        for co in unique_costs:
            class_idx = np.where(classes == cl)[0]
            cost_idx = np.where(cost <= co)[0]
            # Get best hyperparams from cv
            error, portfolio, best_params, cv_portfolio = \
                    outer_cv(x[np.ix_(class_idx, cost_idx)], y[class_idx], m, cv_params)
            # Train model with best hyperparams on full data
            m.set_params(**best_params)
            m.fit(x[np.ix_(class_idx, cost_idx)], y[class_idx])
            # Get human readable details on the selected portfolio
            portfolio_weight, portfolio_name = get_portfolio_details(portfolio, names)
            # Get statistics
            mae, rmsd, maxerr, sample_mae, sample_rmsd, nll_laplace, nll_gaussian = get_error_statistics(error)

            # Store various attributes for the given cost
            params.append(best_params)
            estimators.append(copy.copy(m))
            errors.append(error)
            cv_portfolios.append(cv_portfolio)
            portfolios.append(portfolio)
            portfolio_weights.append(portfolio_weight)
            portfolio_names.append(portfolio_name)
            maes.append(mae)
            rmsds.append(rmsd)
            maxerrs.append(maxerr)
            sample_maes.append(sample_mae)
            sample_rmsds.append(sample_rmsd)
            laplace_mnll.append(nll_laplace)
            gaussian_mnll.append(nll_gaussian)

        # Store various attributes for the given class
        d[cl] = {'errors': errors,
                 'cv_portfolios': cv_portfolios,
                 'portfolios': portfolios,
                 'params': params,
                 'estimators': estimators,
                 'cost': unique_costs,
                 'portfolio_weights': portfolio_weights,
                 'portfolio_names': portfolio_names,
                 'maes': maes,
                 'rmsds': rmsds,
                 'maxerrs': maxerrs,
                 'sample_maes': sample_maes,
                 'sample_rmsds': sample_rmsds,
                 'laplace_mnll': laplace_mnll,
                 'gaussian_mnll': gaussian_mnll
                 }
    
    # Dump the results in a pickle
    with open("pickles/single_method_results.pkl", 'wb') as f:
        pickle.dump(d, f, -1)

def parse_reaction_dataframe(df):
    # just to make sure that stuff is sorted
    # supress warning as this works like intended
    pd.options.mode.chained_assignment = None
    df.sort_values(['functional', 'basis', 'unrestricted', 'reaction'])
    pd.options.mode.chained_assignment = "warn"

    unique_reactions = df.reaction.unique()

    energies = []
    errors = []
    classes = []
    for idx, reac in enumerate(unique_reactions):
        sub_df = df.loc[df.reaction == reac]
        energies.append(sub_df.energy.values)
        errors.append(sub_df.error.values)
        classes.append(sub_df.main_class.values[0])

        # get names of the methods
        if idx == 0:
            func = sub_df.functional.values
            basis = sub_df.basis.values
            unres = ['u-'*int(i) for i in sub_df.unrestricted]
            names = [u + f + "/" + b for b,f,u in zip(basis,func,unres)]


    energies = np.asarray(energies, dtype = float)
    errors = np.asarray(errors, dtype = float)
    classes = np.asarray(classes, dtype = int)
    
    reference = (energies - errors)[:,0]

    # Set the cost to be the biggest reaction
    # This is convoluted since simpler solutions
    # disregard that df might be a subset of another
    # DataFrame
    idx = np.zeros(df.index.max()+1, dtype = int)
    idx[df.index.values] = np.arange(df.index.values.size)
    cost = df[df.reaction == df.iloc[idx[df.time.idxmax()]].reaction].time.values

    return energies, reference, cost, names, classes

def get_portfolio_details(x, names):

    # Get the order by contribution of the portfolio
    idx = np.argsort(x)

    w = []
    n = []
    
    for i in idx:
        weight = x[i]
        if weight < 1e-9:
            continue
        w.append(weight)
        n.append(names[i])

    return w, n



def outer_cv(x, y, m, params, grid = True, 
        outer_cv_splits = 3, outer_cv_repeats = 1, inner_cv_splits = 3, inner_cv_repeats = 1):
    """
    Do outer cross validation to get the prediction errors of a method. 
    """

    if grid:
        cv_model = sklearn.model_selection.GridSearchCV
    else:
        cv_model = sklearn.model_selection.RandomizedSearchCV

    outer_cv_generator = sklearn.model_selection.RepeatedKFold(
            n_splits = outer_cv_splits, n_repeats = outer_cv_repeats)
    inner_cv_generator = sklearn.model_selection.RepeatedKFold(
            n_splits = inner_cv_splits, n_repeats = inner_cv_repeats)

    best_cv_params = []
    cv_portfolios = []
    errors = np.zeros((x.shape[0], outer_cv_repeats))

    for i, (train_idx, test_idx) in enumerate(outer_cv_generator.split(y)):
        train_x, train_y = x[train_idx], y[train_idx]
        test_x, test_y = x[test_idx], y[test_idx]
        if len(params) > 0:
            cvmod = cv_model(m, param_grid = params, scoring = 'neg_mean_absolute_error',
                    return_train_score = False, cv = inner_cv_generator)
            cvmod.fit(train_x, train_y)
            cv_portfolios.append(cvmod.best_estimator_.portfolio)
            best_cv_params.append(cvmod.best_params_)
            y_pred = cvmod.predict(test_x)
        else:
            m.fit(train_x, train_y)
            cv_portfolios.append(m)
            best_cv_params.append(params)
            y_pred = m.predict(test_x)

        errors[test_idx, i // outer_cv_splits] = test_y - y_pred

    # Get the best params
    best_params = get_best_params(best_cv_params)

    # retrain the model on the full data
    m.set_params(**best_params)
    m.fit(x, y)
    final_portfolio = m.portfolio

    # Since the repeats of the same sample is correlated, it's
    # probably better to take the mean of them before doing
    # summary statistics.
    # This means that we're actually using slightly more than (m-1)/m
    # of the data, where m is the number of CV folds.
    # But since we're not doing learning curves, this should be fine.
    errors = np.mean(errors, axis = 1)

    cv_portfolios = np.asarray(cv_portfolios)

    return errors, final_portfolio, best_params, cv_portfolios

def get_best_params(params):
    """
    Attempts to get the best set of parameters
    from a list of dictionaries containing the
    parameters that resulted in the best cv score
    """

    # Preprocess a bit
    d = {}
    for param in params:
        for key,value in param.items():
            if key not in d: d[key] = []
            d[key].append(value)

    # Select the most likely or median value
    for key,value in d.items():
        # if list choose most common
        if isinstance(value[0], str):
            best_value = max(set(value), key=value.count)
            d[key] = best_value
            continue

        # if numeric return median
        d[key] = np.median(value)

    return d


if __name__ == "__main__":
    df = pd.read_pickle("pickles/abde12_reac.pkl")
    # all but x,y is optional
    x, y, cost, names, rclass = parse_reaction_dataframe(df)
    
    m = LinearModel(clip_value = 1e-3, cost = cost, cost_reg = 0)
    z, w = outer_cv(x,y,m,{}, True, 10, 5, 5, 1)[:2]
    print(names[w.argmax()])
    print(np.mean(abs(z)))
    quit()
    m = NN(tensorboard_dir = '', learning_rate = 0.1, iterations = 50000,
            l2_reg = 0, cost_reg = 0, cost = cost)
    m.fit(x,y)
    print(np.sort(m.portfolio)[-5:])
    print(names[np.argmax(m.portfolio)])

    #m = NN(tensorboard_dir = 'log', learning_rate = 0.001, iterations = 1000000, 
    #        l2_reg = 1e-6, cost_reg = 1e-9, cost = cost)
    #m.fit(x,y)
    #quit()
    #p = np.where(m.portfolio > 0.01)[0]
    #out = df[df.reaction == df.iloc[df.time.idxmax()].reaction].values[p][:,[0,3,-2]]
    #print(np.concatenate([out,m.portfolio[p,None]], axis=1))

    #run_SingleMethod(x,y, cost, names, rclass)
    #run_linear(x,y, cost, names, None)

    #def __init__(self, learning_rate = 0.3, iterations = 5000, cost_reg = 0.0, l2_reg = 0.0, 
    #        scoring_function = 'rmse', optimiser = "Adam", softmax = True, fit_bias = False,
    #        nhl = 0, hl1 = 5, hl2 = 5, hl3 = 5, multiplication_layer = False, activation_function = "sigmoid",
    #        bias_input = False, n_main_features = -1, single_thread = True, tensorboard_dir = '', 
    #        tensorboard_store_frequency = 100, **kwargs):
