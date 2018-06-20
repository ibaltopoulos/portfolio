"""
Generate data for paper

"""

import os, sys
import pandas as pd
import numpy as np
import sklearn.model_selection

# Add the module to source path
dirname = os.path.dirname(os.path.realpath(__file__))
source_path = dirname + "/portfolio"
sys.path.append(source_path)

from portfolio.model import SingleMethod, NN

def run_SingleMethod(x,y, cost = None, names = None):
    m = SingleMethod(loss = 'mae')
    score = outer_cv(x, y, m)
    print("SingleMethod score:", score, loss, scoring_function)
    return score

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

# TODO this can be done with a pipeline
def outer_cv(x, y, m):
    """
    Do outer cross validation to get the prediction errors of a method. 
    kwargs are a dictionary with options to the Portfolio class.
    """

    #from sklearn.model_selection import cross_val_score, cross_validate
    #scores = cross_validate(m, x, y, cv=3)
    #print(scores)

    #quit()

    cv_generator = sklearn.model_selection.RepeatedKFold(n_splits = 5, n_repeats = 10000)

    errors = []
    for train_idx, test_idx in cv_generator.split(y):
        train_x, train_y = x[train_idx], y[train_idx]
        test_x, test_y = x[test_idx], y[test_idx]

        m.fit(train_x, train_y)
        errors.append(m.score(test_x, test_y))

    return np.mean(errors)

if __name__ == "__main__":
    df = pd.read_pickle(sys.argv[1])
    # all but x,y is optional
    x, y, cost, names, rclass = parse_reaction_dataframe(df)

    #m = NN(tensorboard_dir = 'log', learning_rate = 0.001, iterations = 1000000, 
    #        l2_reg = 1e-6, cost_reg = 1e-9, cost = cost)
    #m.fit(x,y)
    #quit()
    #p = np.where(m.portfolio > 0.01)[0]
    #out = df[df.reaction == df.iloc[df.time.idxmax()].reaction].values[p][:,[0,3,-2]]
    #print(np.concatenate([out,m.portfolio[p,None]], axis=1))

    run_SingleMethod(x,y)
    #run_linear(x,y, cost, names, None)

    #def __init__(self, learning_rate = 0.3, iterations = 5000, cost_reg = 0.0, l2_reg = 0.0, 
    #        scoring_function = 'rmse', optimiser = "Adam", softmax = True, fit_bias = False,
    #        nhl = 0, hl1 = 5, hl2 = 5, hl3 = 5, multiplication_layer = False, activation_function = "sigmoid",
    #        bias_input = False, n_main_features = -1, single_thread = True, tensorboard_dir = '', 
    #        tensorboard_store_frequency = 100, **kwargs):
