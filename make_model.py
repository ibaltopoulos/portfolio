from neural_net import SingleLayeredNeuralNetwork
import numpy as np
import pickle

m = SingleLayeredNeuralNetwork(learning_rate = 1e-1, n_hidden = 20, iterations = 20000)

x = np.random.random((1000,50))
a = np.random.random(50)
a /= a.sum()
y = np.sum(x * a, 1)

d = {'x': x, 'y': y}
with open('data.pkl', 'wb') as f:
    pickle.dump(d, f, -1)
with open('model.pkl', 'wb') as f:
    pickle.dump(m, f)
