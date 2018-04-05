
from __future__ import print_function
import numpy as np
from sklearn.base import BaseEstimator
import tensorflow as tf
from utils import is_positive_or_zero, is_positive, is_positive_integer, \
    is_string, InputError
from inspect import signature
from sklearn.utils.validation import check_X_y, check_array
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def is_positive(x):
    return (not is_array_like(x) and _is_numeric(x) and x > 0)

def is_positive_or_zero(x):
    return (not is_array_like(x) and _is_numeric(x) and x >= 0)

def is_array_like(x):
    return isinstance(x, (tuple, list, np.ndarray))

def is_positive_integer(x):
    return (not is_array_like(x) and _is_integer(x) and x > 0)

def is_string(x):
    return isinstance(x, str)

def _is_numeric(x):
    return isinstance(x, (float, int))

def _is_integer(x):
    return (_is_numeric(x) and (float(x) == int(x)))

# custom exception to raise when we intentionally catch an error
class InputError(Exception):
    pass


class Osprey(BaseEstimator):
    """
    Overrides scikit-learn calls to make inheritance work in the 
    Osprey bayesian optimisation package.
    """

    def __init__(self, **kwargs):
        pass

    def get_params(self, deep = True):
        """
        Hack that overrides the get_params routine of BaseEstimator.
        self.get_params() returns the input parameters of __init__. However it doesn't
        handle inheritance well, as we would like to include the input parameters to
        __init__ of all the parents as well.

        """
        import inspect
        from optparse import OptionParser
        from inspect import getargvalues, stack
        import ast

        # First get the name of the class self belongs to
        name_of_class = self.__class__.__name__

        # Then get names of the parents and their parents etc
        # excluding 'object'
        names_of_parents = [c.__name__ for c in eval(name_of_class + ".__bases__") if c.__name__ not in "object"]
        # limit to 10 generations to avoid infinite loop
        for i in range(10):
            new_parents = False
            for name in names_of_parents:
                new_names_of_parents = [c.__name__ for c in eval(name + ".__bases__")]
                for new_name in new_names_of_parents:
                    if new_name not in names_of_parents and new_name != "object":
                        names_of_parents.append(new_name)
                        new_parents = True
            if not new_parents:
                break
        else:
            print("Warning: Only included first 10 generations of parents of the called class")

        params = BaseEstimator.get_params(self)
        for parent in names_of_parents:
            parent_init = eval(parent + ".__init__")

            # Modified from the scikit-learn BaseEstimator class
            parent_init_signature = signature(parent_init)
            for p in (p for p in parent_init_signature.parameters.values() 
                    if p.name != 'self' and p.kind != p.VAR_KEYWORD):
                if p.name in params:
                    raise InputError('This should never happen')
                if hasattr(self, p.name):
                    params[p.name] = getattr(self, p.name)
                else:
                    params[p.name] = p.default

        return params

    def set_params(self, **params):
        """
        Hack that overrides the set_params routine of BaseEstimator.

        """
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)

        return self

    def score(self, x, y = None):
        # Osprey maximises a score per default, so return minus mae/rmsd and plus r2
        if self.scoring_function == "r2":
            return self._score(x, y)
        else:
            return - self._score(x, y)

class _NN(object):
    """
    Base class for neural network stuff.
    Children should overwrite the _model, _generate_weights and _cost functions.

    """

    def __init__(self, learning_rate = 0.01, iterations = 500, l1_reg = 0.0, l2_reg = 0.0, 
            scoring_function = 'rmse', early_stopping = True, **kwargs):
        """
        :param l1_reg: L1-regularisation parameter for the neural network weights
        :type l1_reg: float
        :param l2_reg: L2-regularisation parameter for the neural network weights
        :type l2_reg: float
        :param learning_rate: The learning rate in the numerical minimisation.
        :type learning_rate: float
        :param iterations: Total number of iterations that will be carried out during the training process.
        :type iterations: integer
        :param scoring_function: Scoring function to use. Available choices are 'mae', 'rmse', 'r2'.
        :type scoring_function: string
        """

        self._set_l1_reg(l1_reg)
        self._set_l2_reg(l2_reg)
        self._set_learning_rate(learning_rate)
        self._set_iterations(iterations)
        self._set_scoring_function(scoring_function)

        # Placeholder variables
        self.n_features = None
        self.n_samples = None
        self.training_cost = []
        self.session = None

    def _set_l1_reg(self, l1_reg):
        if not is_positive_or_zero(l1_reg):
            raise InputError("Expected positive float value for variable 'l1_reg'. Got %s" % str(l1_reg))
        self.l1_reg = l1_reg

    def _set_l2_reg(self, l2_reg):
        if not is_positive_or_zero(l2_reg):
            raise InputError("Expected positive float value for variable 'l2_reg'. Got %s" % str(l2_reg))
        self.l2_reg = l2_reg

    def _set_learning_rate(self, learning_rate):
        if not is_positive(learning_rate):
            raise InputError("Expected positive float value for variable learning_rate. Got %s" % str(learning_rate))
        self.learning_rate = float(learning_rate)

    def _set_iterations(self, iterations):
        if not is_positive_integer(iterations):
            raise InputError("Expected positive integer value for variable iterations. Got %s" % str(iterations))
        self.iterations = int(iterations)

    def _set_scoring_function(self, scoring_function):
        if not is_string(scoring_function):
            raise InputError("Expected a string for variable 'scoring_function'. Got %s" % str(scoring_function))
        if scoring_function.lower() not in ['mae', 'rmse', 'r2']:
            raise InputError("Available scoring functions are 'mae', 'rmse', 'r2'. Got %s" % str(scoring_function))

        self.scoring_function = scoring_function

    def _l2_loss(self, weights):
        """
        Creates the expression for L2-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list or tensor
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        reg_term = tf.zeros([])

        if isinstance(weights, list):
            for i in range(len(weights)):
                reg_term += tf.reduce_sum(tf.square(weights[i]))
        else:
            reg_term += tf.reduce_sum(tf.square(weights))

        return self.l2_reg * reg_term

    def _l1_loss(self, weights):
        """
        Creates the expression for L1-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list or tensor
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        reg_term = tf.zeros([])

        if isinstance(weights, list):
            for i in range(len(weights)):
               reg_term += tf.reduce_sum(tf.abs(weights[i]))
        else:
            reg_term += tf.reduce_sum(tf.abs(weights))

        return self.l1_reg * reg_term

    def plot_cost(self, filename = None):
        """
        Plots the value of the cost function as a function of the iterations.

        :param filename: File to save the plot to. If '' the plot is shown instead of saved.
        :type filename: string
        """

        try:
            import pandas as pd
            import seaborn as sns
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Plotting functions require the modules 'seaborn' and 'pandas'")

        sns.set()
        df = pd.DataFrame()
        df["Iterations"] = range(len(self.training_cost))
        df["Training cost"] = self.training_cost
        f = sns.lmplot('Iterations', 'Training cost', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5}, fit_reg=False)
        f.set(yscale = "log")

        if filename == None:
            plt.show()
        elif is_string(filename):
            plt.save(filename)
        else:
            raise InputError("Wrong data type of variable 'filename'. Expected string")

    def correlation_plot(self, y_nn, y_true, filename = ''):
        """
        Creates a correlation plot between predictions and true values.

        :param y_predicted: Values predicted by the neural net
        :type y_predicted: list
        :param y_true: True values
        :type y_true: list
        :param filename: File to save the plot to. If '' the plot is shown instead of saved.
                         If the dimensionality of y is higher than 1, the filename will be prefixed
                         by the dimension.
        :type filename: string
        """

        try:
            import pandas as pd
            import seaborn as sns
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Plotting functions require the modules 'seaborn' and 'pandas'")

        if y_nn.shape != y_true.shape:
            raise InputError("Shape mismatch between predicted and true values. %s and %s" % (str(y_nn.shape), str(y_true.shape)))

        if y_nn.ndim == 1 or y_nn.shape[1] == 1:
            df = pd.DataFrame()
            df["Predictions"] = y_nn.ravel()
            df["True"] = y_true.ravel()
            sns.set()
            lm = sns.lmplot('True', 'Predictions', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5})
            if filename == '':
                plt.show()
            elif is_string(filename):
                plt.save(filename)
            else:
                raise InputError("Wrong data type of variable 'filename'. Expected string")
        else:
            for i in range(y_nn.shape[0]):
                df = pd.DataFrame()
                df["Predictions"] = y_nn[:,i]
                df["True"] = y_true[:,i]
                sns.set()
                lm = sns.lmplot('True', 'Predictions', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5})
                if filename == '':
                    plt.show()
                elif is_string(filename):
                    tokens = filename.split("/")
                    file_ = str(i) + "_" + tokens[-1]
                    if len(tokens) > 1:
                        file_ = "/".join(tokens[:-1]) + "/" + file_
                    plt.save(file_)
                else:
                    raise InputError("Wrong data type of variable 'filename'. Expected string")

    def _score(self, *args):
        if self.scoring_function == 'mae':
            return self._score_mae(*args)
        if self.scoring_function == 'rmse':
            return self._score_rmse(*args)
        if self.scoring_function == 'r2':
            return self._score_r2(*args)

    def predict(self, x):
        predictions = self._predict(x)

        if predictions.ndim > 1 and predictions.shape[1] == 1:
            return predictions.ravel()
        else:
            return predictions

    def _predict(self, x):
        """
        Use the trained network to make predictions on the data x.

        :param x: The input data of shape (n_samples, n_features)
        :type x: array

        :return: Predictions for the target values corresponding to the samples contained in x.
        :rtype: array

        """

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        check_array(x, warn_on_dtype = True)

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("x:0")
            model = graph.get_tensor_by_name("y:0")
            y_pred = self.session.run(model, feed_dict = {tf_x : x})
            return y_pred

    def fit(self, x, y):
        """
        Fit the neural network to input x and target y.

        :param x: Input data 
        :type x: array of size (n_samples, n_features)
        :param y: Target values
        :type y: array of size (n_samples, )

        """

        # Check that X and y have correct shape
        x, y = check_X_y(x, y, multi_output = False, y_numeric = True, warn_on_dtype = True)

        # reshape to tensorflow friendly shape
        y = np.atleast_2d(y).T

        # Collect size input
        self.n_features = x.shape[1]
        self.n_samples = x.shape[0]

        # Initial set up of the NN
        tf_x = tf.placeholder(tf.float32, [None, self.n_features], name="x")
        tf_y = tf.placeholder(tf.float32, [None, 1])

        # Generate weights
        weights, biases = self._generate_weights()

        # Create the graph
        y_pred = self._model(tf_x, weights, biases)

        cost = self._cost(y_pred, tf_y, weights)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Initialisation of the variables
        init = tf.global_variables_initializer()

        # Uncomment to use all cpus
        session_conf = tf.ConfigProto(
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
        sess = tf.Session(config=session_conf)

        self.session = tf.Session(config = session_conf)

        # Running the graph
        self.session.run(init)

        last_cost = np.inf
        for i in range(self.iterations):
            feed_dict = {tf_x: x, tf_y: y}
            opt, avg_cost = self.session.run([optimizer, cost], feed_dict=feed_dict)
            self.training_cost.append(avg_cost)
            if i % 100 == 0:
                if (last_cost - avg_cost) < 1e-4:
                    print("Stopped at iteration", i)
                    break
                last_cost = avg_cost

    def _cost(self, y_pred, tf_y, weights):
        """
        To be overwritten by child methods.
        Takes the model tensor, target tensor and the weights.

        """

        raise NotImplementedError("self._cost should be overwritten by child methods")

    def _model(self, x, weights):
        """
        To be overwritten by child methods.
        Takes input tensor and weights and returns the model.

        """

        raise NotImplementedError("self._model should be overwritten by child methods")

    def _generate_weights(self):
        """
        To be overwritten by child methods.
        Should return the weights and biases.

        """

        raise NotImplementedError("self._generate_weights should be overwritten by child methods")

    def _init_weight(self, n1, n2):
        """
        Generate a tensor of weights of size (n1, n2)

        """

        w = tf.Variable(tf.truncated_normal([n1,n2], stddev = 1.0 / np.sqrt(n2)))

        return w

    def _init_bias(self, n):
        """
        Generate a tensor of biases of size n.

        """

        b = tf.Variable(tf.zeros([n], dtype = tf.float32))

        return b

    def _score_r2(self, x, y, sample_weight=None):
        """
        Calculate the coefficient of determination (R^2).
        Larger values corresponds to a better prediction.

        :param x: The input data.
        :type x: array of shape (n_samples, n_features)
        :param y: The target values for each sample in x.
        :type y: array of shape (n_samples,)

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: R^2
        :rtype: float

        """

        y_pred = self.predict(x)
        r2 = r2_score(y, y_pred, sample_weight = sample_weight)
        return r2

    def _score_mae(self, x, y, sample_weight=None):
        """
        Calculate the mean absolute error.
        Smaller values corresponds to a better prediction.

        :param x: The input data.
        :type x: array of shape (n_samples, n_features)
        :param y: The target values for each sample in x.
        :type y: array of shape (n_samples,)

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: Mean absolute error
        :rtype: float

        """

        y_pred = self.predict(x)
        mae = mean_absolute_error(y, y_pred, sample_weight = sample_weight)
        return mae

    def _score_rmse(self, x, y, sample_weight=None):
        """
        Calculate the root mean squared error.
        Smaller values corresponds to a better prediction.

        :param x: The input data.
        :type x: array of shape (n_samples, n_features)
        :param y: The target values for each sample in x.
        :type y: array of shape (n_samples,)

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: Mean absolute error
        :rtype: float

        """

        y_pred = self.predict(x)
        rmse = np.sqrt(mean_squared_error(y, y_pred, sample_weight = sample_weight))
        return rmse


class ConstrainedElasticNet(_NN, Osprey):
    """
    Solves a elastic net under the constraints that the weights
    sum to one and are all positive (softmax).
    l1 regression doesn't do anything due to the constraints,
    but l2 regression can still be included.

    """

    def __init__(self, **kwargs):
        super(ConstrainedElasticNet, self).__init__(**kwargs)

    def _generate_weights(self):
        """
        Generates the weights.

        :return: tuple of weights and biases, however there are no biases in this model.
        :rtype: tuple
        """

        weights = self._init_weight(self.n_features, 1)

        return weights, None

    def _model(self, x, weights, biases = None):
        """
        Constructs the actual network.

        :param x: Input
        :type x: tf.placeholder of shape (None, n_features)
        :param weights: Weights used in the network.
        :type weights: tf.Variable of shape (n_features, 1)
        :param biases: Dummy variable
        :type weights: NoneType
        :return: Output
        :rtype: tf.Variable of size (None, n_targets)
        """

        # Softmax activation function
        h = tf.nn.softmax(weights, dim = 0)
        # Add up contributions (name must be 'y')
        z = tf.matmul(x, h, name = 'y')

        return z

    def _cost(self, y_pred, y, weights):
        """
        Constructs the cost function

        :param y_pred: Predicted output
        :type y_pred: tf.Variable of size (None, 1)
        :param y: True output
        :type y: tf.placeholder of shape (None, 1)
        :param weights: Weights used in the network.
        :type weights: tf.Variable of shape (n_features, 1)
        :return: Cost
        :rtype: tf.Variable of size (1,)
        """

        err = tf.square(tf.subtract(y,y_pred))
        loss = tf.reduce_mean(err)
        cost = loss
        if self.l2_reg > 0:
            l2_loss = self._l2_loss(weights)
            cost = cost + l2_loss

        return cost


class SingleLayeredNeuralNetwork(_NN, Osprey):
    """
    Neural net with a single hidden layer.
    Hyper-parameters are l1_reg, l2_reg, n_hidden

    """

    def __init__(self, n_hidden = 5, activation_function = "sigmoid", **kwargs):
        super(SingleLayeredNeuralNetwork, self).__init__(**kwargs)

        self.n_hidden = n_hidden
        self._set_activation_function(activation_function)

    def _set_activation_function(self, activation_function):
        if activation_function in ['sigmoid', tf.nn.sigmoid]:
            self.activation_function = tf.nn.sigmoid
        elif activation_function in ['tanh', tf.nn.tanh]:
            self.activation_function = tf.nn.tanh
        elif activation_function in ['elu', tf.nn.elu]:
            self.activation_function = tf.nn.elu
        elif activation_function in ['softplus', tf.nn.softplus]:
            self.activation_function = tf.nn.softplus
        elif activation_function in ['softsign', tf.nn.softsign]:
            self.activation_function = tf.nn.softsign
        elif activation_function in ['relu', tf.nn.relu]:
            self.activation_function = tf.nn.relu
        elif activation_function in ['relu6', tf.nn.relu6]:
            self.activation_function = tf.nn.relu6
        elif activation_function in ['crelu', tf.nn.crelu]:
            self.activation_function = tf.nn.crelu
        elif activation_function in ['relu_x', tf.nn.relu_x]:
            self.activation_function = tf.nn.relu_x
        else:
            raise InputError("Unknown activation function. Got %s" % str(activation_function))

    def _generate_weights(self):
        """
        Generates the weights.

        :return: tuple of weights and biases
        :rtype: tuple
        """

        weights, biases = [], []

        weights.append(self._init_weight(self.n_features, self.n_hidden))
        weights.append(self._init_weight(self.n_hidden, 1))

        biases.append(self._init_bias(self.n_hidden))
        biases.append(self._init_bias(1))

        return weights, biases


    def _model(self, x, weights, biases = None):
        """
        Constructs the actual network.

        :param x: Input
        :type x: tf.placeholder of shape (None, n_features)
        :param weights: Weights used in the network.
        :type weights: list of tf.Variables of shape (n_features, n_hidden)
            and (n_hidden, 1)
        :param biases: Biases used in the network
        :type weights: list of tf.Variables of shape (n_hidden,) and (1,)
        :return: Output
        :rtype: tf.Variable of size (None, n_targets)
        """

        # get activations of hidden layer
        z = tf.add(tf.matmul(x, weights[0]), biases[0])
        h = self.activation_function(z)

        # Add up contributions (name must be 'y')
        z = tf.add(tf.matmul(h, weights[1]), biases[1], name = 'y')

        return z

    def _cost(self, y_pred, y, weights):
        """
        Constructs the cost function

        :param y_pred: Predicted output
        :type y_pred: tf.Variable of size (None, 1)
        :param y: True output
        :type y: tf.placeholder of shape (None, 1)
        :param weights: Weights used in the network.
        :type weights: list of tf.Variable
        :return: Cost
        :rtype: tf.Variable of size (1,)
        """

        err = tf.square(tf.subtract(y,y_pred))
        loss = tf.reduce_mean(err)
        cost = loss
        if self.l2_reg > 0:
            l2_loss = self._l2_loss(weights)
            cost = cost + l2_loss
        if self.l1_reg > 0:
            l1_loss = self._l1_loss(weights)
            cost = cost + l1_loss

        return cost


if __name__ == "__main__":
    np.random.seed(42)
    #m = ConstrainedElasticNet(learning_rate = 1e1, iterations = 100)
    m = SingleLayeredNeuralNetwork(learning_rate = 1e-1, n_hidden = 20, iterations = 10000, l2_reg = 1e-3)

    x = np.random.random((1000,50))
    a = np.random.random(50)
    a /= a.sum()
    y = np.sum(x * a, 1)

    m.fit(x[:800],y[:800])

    y_pred = m.predict(x[800:])

    plt.scatter(y[800:], y_pred)
    plt.show()

    score = m.score(x[800:], y[800:])
    print(score)


