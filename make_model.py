from neural_net import ConstrainedElasticNet, reaction_dataframe_to_energies
import numpy as np
import pickle
import sys
import pandas as pd

m = ConstrainedElasticNet(iterations = 10000)

df = pd.read_pickle(sys.argv[1])
x, y = reaction_dataframe_to_energies(df)

d = {'x': x, 'y': y}
with open('data.pkl', 'wb') as f:
    pickle.dump(d, f, -1)
with open('model.pkl', 'wb') as f:
    pickle.dump(m, f)
