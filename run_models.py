"""
Generate data for paper

"""

import os, sys
import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection

# Add the module to source path
dirname = os.path.dirname(os.path.realpath(__file__))
source_path = dirname + "/portfolio"
sys.path.append(source_path)

from portfolio.model import SingleMethod, NN

def run_SingleMethod(x,y, cost = None, names = None, classes = None):
    if os.path.isfile("pickles/single_method_results.pkl"):
        return
    
    m = SingleMethod()
    params = {'loss': ('mae', 'rmsd', 'max')}
    # Separate different classes
    unique_classes = np.unique(classes)
    for c in unique_classes:
        idx = np.where(classes == c)[0]
    errors, best_cv_params, cv_portfolios, final_portfolio, final_params = \
            outer_cv(x[idx], y[idx], m, params)

    d = {'errors': errors,
         'cv_params': best_cv_params,
         'cv_portfolios': cv_portfolios,
         'final_portfolio': final_portfolio,
         'final_params': final_params}

def run_LinearModel(x,y, cost = None, names = None):
    m = NN(tensorboard_dir = '', learning_rate = 1e-1, iterations = 10000, 
            l2_reg = 1e-6, cost_reg = 1e-9, cost = cost, lol = 2)
    score = outer_cv(x, y, m)
    print("Linear Model score:", score)
    return score

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
    cost = df[df.reaction == df.iloc[df.time.idxmax()].reaction].time.values


    return energies, reference, cost, names, classes

def outer_cv(x, y, m, params, grid = True):
    """
    Do outer cross validation to get the prediction errors of a method. 
    """

    if grid:
        cv_model = sklearn.model_selection.GridSearchCV
    else:
        cv_model = sklearn.model_selection.RandomizedSearchCV

    outer_cv_splits = 5
    outer_cv_repeats = 1
    inner_cv_splits = 3
    inner_cv_repeats = 1

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
        cvmod = cv_model(m, param_grid = params, scoring = 'neg_mean_absolute_error',
                return_train_score = False, cv = inner_cv_generator)
        cvmod.fit(train_x, train_y)
        cv_portfolios.append(cvmod.best_estimator_.portfolio)
        best_cv_params.append(cvmod.best_params_)
        y_pred = cvmod.predict(test_x)
        errors[test_idx, i // outer_cv_splits] = test_y - y_pred

    best_params = get_best_params(best_cv_params)

    m.set_params(**best_params)
    m.fit(x, y)
    final_portfolio = m.portfolio

    return errors, best_cv_params, cv_portfolios, final_portfolio, best_params

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

    #m = NN(tensorboard_dir = 'log', learning_rate = 0.001, iterations = 1000000, 
    #        l2_reg = 1e-6, cost_reg = 1e-9, cost = cost)
    #m.fit(x,y)
    #quit()
    #p = np.where(m.portfolio > 0.01)[0]
    #out = df[df.reaction == df.iloc[df.time.idxmax()].reaction].values[p][:,[0,3,-2]]
    #print(np.concatenate([out,m.portfolio[p,None]], axis=1))

    run_SingleMethod(x,y, cost, names, rclass)
    #run_linear(x,y, cost, names, None)

    #def __init__(self, learning_rate = 0.3, iterations = 5000, cost_reg = 0.0, l2_reg = 0.0, 
    #        scoring_function = 'rmse', optimiser = "Adam", softmax = True, fit_bias = False,
    #        nhl = 0, hl1 = 5, hl2 = 5, hl3 = 5, multiplication_layer = False, activation_function = "sigmoid",
    #        bias_input = False, n_main_features = -1, single_thread = True, tensorboard_dir = '', 
    #        tensorboard_store_frequency = 100, **kwargs):
