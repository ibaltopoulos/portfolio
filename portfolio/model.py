"""
Methods to create portfolios
"""

from __future__ import print_function
import pickle, sys, os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from inspect import signature
from .utils import InputError, is_string, is_positive_or_zero, is_positive, \
        is_positive_integer, is_positive_integer_or_zero, is_bool, is_none, \
        is_positive_array
from .tf_utils import TensorBoardLogger


class BaseModel(BaseEstimator):
    """
    Base class for all predictive models.

    """
    def __init__(self, scoring_function = 'root_mean_squared_error', **kwargs):
        self._set_scoring_function(scoring_function)
        # Placeholder variables
        self.n_features = None
        self.n_samples = None

        # There should be no arguments that is not named
        if kwargs != {}:
            raise InputError("Got unknown input %s to the class %s" % (kwargs, self.__class__.__name__))

    def _set_scoring_function(self, scoring_function):
        if not is_string(scoring_function):
            raise InputError("Expected a string for variable 'scoring_function'. Got %s" % str(scoring_function))
        if scoring_function.lower() not in ['mean_absolute_error', 'root_mean_squared_error', 'maximum_absolute_error',
            'negative_mean_absolute_error', 'negative_mean_squared_error', 'negative_maximum_absolute_error']:
            raise InputError("Unknown scoring functions '%s'" % str(scoring_function))

        self.scoring_function = scoring_function

    def predict(self, x):
        raise NotImplementedError

    def fit(self, x, y):
        raise NotImplementedError

    def score(self, x, y, sample_weight = None):

        y_pred = self.predict(x)

        # Do it this way so that we can always extract multiple scores
        self.score_mae = mean_absolute_error(y_pred, y, sample_weight)
        self.score_rmsd = np.sqrt(mean_squared_error(y_pred, y, sample_weight))
        self.score_max = max(abs(y_pred - y))

        if self.scoring_function == 'mean_absolute_error':
            return self.score_mae
        elif self.scoring_function == 'root_mean_squared_error':
            return self.score_rmsd
        elif self.scoring_function == 'maximum_absolute_error':
            return self.score_max
        elif self.scoring_function == 'negative_mean_absolute_error':
            return - self.score_mae
        elif self.scoring_function == 'negative_mean_squared_error':
            return - self.score_rmsd
        elif self.scoring_function == 'negative_maximum_absolute_error':
            return - self.score_max


class NN(BaseModel):
    """
    Neural network predictor.

    """

    def __init__(self, learning_rate = 0.3, iterations = 5000, cost_reg = 0.0, l2_reg = 0.0, 
            optimiser = "Adam", softmax = True, fit_bias = False,
            nhl = 0, hl1 = 5, hl2 = 5, hl3 = 5, multiplication_layer = False, activation_function = "sigmoid",
            bias_input = False, n_main_features = -1, single_thread = True, tensorboard_dir = '', 
            tensorboard_store_frequency = 100, cost = None, **kwargs):
        """
        :param learning_rate: The learning rate in the numerical minimisation.
        :type learning_rate: float
        :param iterations: Total number of iterations that will be carried out during the training process.
        :type iterations: integer
        :param cost_reg: L1-regularisation parameter on the cost for the neural network
        :type cost_reg: float
        :param l2_reg: L2-regularisation parameter for the neural network weights
        :type l2_reg: float
        :param scoring_function: Scoring function to use. Available choices are `mae`, `rmse`, `r2`.
        :type scoring_function: string
        :param optimiser: Which tensorflow optimiser to use
        :type optimiser: string or tensorflow optimizer
        :param softmax: Use softmax on the method (portfolio) weights, such that all weights are positive and sum to one.
        :type softmax: bool
        :param fit_bias: Fit a bias to the final portfolio to offset systematic errors
        :type fit_bias: bool
        :param nhl: Number of hidden layers. Has to be between 1 and 3.
        :type nhl: int
        :param hl1: Size of first hidden layer
        :type hl1: int
        :param hl2: Size of first hidden layer
        :type hl2: int
        :param hl3: Size of first hidden layer
        :type hl3: int
        :param multiplication_layer: Forces that the final result is a linear combination of the main_features
        :type multiplication_layer: bool
        :param activation_function: Activation function of the hidden layers.
        :type activation_function: string
        :param bias_input: Subtract a weighted mean from the main input features
        :type bias_input: bool
        :param n_main_features: The number of main features
        :type n_main_features: int
        :param single_thread: Force tensorflow to use only one thread. Should be False for gpus
        :type single_thread: bool
        :param tensorboard_dir: Directory for tensorboard logging. Logging won't be performed if `tensorboard_dir = ''` 
        :type tensorboard_dir: string
        :param tensorboard_store_frequency: How often to store status in tensorboard
        :type tensorboard_store_frequency: int
        :param cost: Computational cost of the main features
        :type cost: array

        """

        # Initialise parents
        super(self.__class__, self).__init__(**kwargs)
        self._set_cost_reg(cost_reg)
        self._set_l2_reg(l2_reg)
        self._set_learning_rate(learning_rate)
        self._set_iterations(iterations)
        self._set_optimiser(optimiser)
        self._set_softmax(softmax)
        self._set_fit_bias(fit_bias)
        self._set_multiplication_layer(multiplication_layer)
        self._set_activation_function(activation_function)
        self._set_bias_input(bias_input)
        self._set_n_main_features(n_main_features)
        self._set_hidden_layers(nhl, hl1, hl2, hl3)
        self._set_single_thread(single_thread)
        self._set_tensorboard(tensorboard_dir, tensorboard_store_frequency)
        self._set_cost(cost)

        self._validate_options()

        # Placeholder variables
        self.session = None


    def _validate_options(self):
        """
        Checks if there are invalid combinations of options
        """

        if self.softmax and not self.multiplication_layer:
            if self.n_main_features != -1 or self.nhl != 0:
                raise InputError("multiplication_layer can't be False if softmax is True, \
                        unless nhl is 0 and n_features equals n_main_features")

    def _set_hidden_layers(self, nhl, hl1, hl2, hl3):
        if is_positive_integer_or_zero(nhl) and nhl <= 3:
            self.nhl = nhl
        else:
            raise InputError("Expected variable 'nhl' to be integer and between 0 and 3. Got %s" % str(nhl))

        if is_positive_integer_or_zero(hl1) and \
                is_positive_integer_or_zero(hl2) and \
                is_positive_integer_or_zero(hl3):

            self.hl1 = int(hl1)
            self.hl2 = int(hl2)
            self.hl3 = int(hl3)
        else:
            raise InputError("Expected variable 'nhl' to be integer and between 0 and 3. Got %s" % str(nhl))

    def _set_softmax(self, softmax):
        if softmax in [True, False]:
            self.softmax = softmax
        else:
            raise InputError("Expected variable 'softmax' to be boolean. Got %s" % str(softmax))

    def _set_fit_bias(self, fit_bias):
        if fit_bias in [True, False]:
            self.fit_bias = fit_bias
        else:
            raise InputError("Expected variable 'fit_bias' to be boolean. Got %s" % str(fit_bias))

    def _set_multiplication_layer(self, multiplication_layer):
        if multiplication_layer in [True, False]:
            self.multiplication_layer = multiplication_layer
        else:
            raise InputError("Expected variable 'multiplication_layer' to be boolean. Got %s" % str(multiplication_layer))

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

    def _set_bias_input(self, bias_input):
        if bias_input in [True, False]:
            self.bias_input = bias_input
        else:
            raise InputError("Expected variable 'bias_input' to be boolean. Got %s" % str(bias_input))

    def _set_n_main_features(self, n_main_features):
        if is_positive_integer(n_main_features) or n_main_features == -1:
            self.n_main_features = n_main_features
        else:
            raise InputError("Expected variable 'n_main_features' to be positive integer. Got %s" % str(n_main_features))

    def _set_cost_reg(self, cost_reg):
        if not is_positive_or_zero(cost_reg):
            raise InputError("Expected positive float value for variable 'cost_reg'. Got %s" % str(cost_reg))
        self.cost_reg = cost_reg

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


    def _set_optimiser(self, optimiser):
        try:
            optimiser = optimiser().get_name()
        except TypeError:
            pass

        if is_string(optimiser):
            if optimiser in ["GradientDescent", "Adadelta", "Adagrad", "Adam", "RMSProp"]:
                self.optimiser = eval("tf.train.%sOptimizer" % optimiser)
        else:
            raise InputError("Expected a string or tensorflow.optimiser object for variable 'optimiser'. Got %s" % str(optimiser))

    def _set_single_thread(self, single_thread):
        if not is_bool(single_thread):
            raise InputError("Expected boolean for variable single_thread. Got %s" % str(single_thread))
        self.single_thread = bool(single_thread)

    def _set_tensorboard(self, tensorboard_dir, store_frequency):

        if tensorboard_dir in ['', None]:
            self.tensorboard_logger = TensorBoardLogger(use_logger = False)
            return

        if not is_string(tensorboard_dir):
            raise InputError('Expected string value for variable tensorboard_dir. Got %s' % str(tensorboard_dir))

        if not is_positive_integer(store_frequency):
            raise InputError("Expected positive integer value for variable store_frequency. Got %s" % str(store_frequency))

        if store_frequency > self.iterations:
            print("Only storing final iteration for tensorboard")
            store_frequency = self.iterations

        # TensorBoardLogger will handle all tensorboard related things
        self.tensorboard_logger = TensorBoardLogger(path = tensorboard_dir, store_frequency = store_frequency)

    def _set_cost(self, cost):
        if is_none(cost):
            self.cost = None
            return
        elif not is_positive_array(cost):
            raise InputError("Expected boolean for variable single_thread. Got %s" % str(single_thread))

        self.cost = np.asarray(cost, dtype = float)

    def _l2_loss(self, weights):
        """
        Creates the expression for L2-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list or tensor
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        with tf.name_scope("l2_reg_loss"):
            reg_term = tf.nn.l2_loss(weights[0])
            for weight in weights[1:]:
                reg_term += tf.nn.l2_loss(weight)
            reg_term *= self.l2_reg

        return reg_term

    def _cost_loss(self):
        """
        Creates the expression for L1-regularisation on the cost

        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        with tf.name_scope("cost_reg_loss"):
            cost = tf.constant(self.cost, shape = (1, self.n_main_features), dtype = tf.float32, name = "comp_cost")
            reg_term = tf.multiply(
                    tf.squeeze(
                        tf.matmul(cost, self.portfolio_weights), name = "estimated_cost"), self.cost_reg, name = "cost_reg")

        return reg_term

#    def plot_loss(self, filename = None):
#        """
#        Plots the value of the loss function as a function of the iterations.
#
#        :param filename: File to save the plot to. If None the plot is shown instead of saved.
#        :type filename: string
#        """
#
#        try:
#            import pandas as pd
#            import seaborn as sns
#        except ModuleNotFoundError:
#            raise ModuleNotFoundError("Plotting functions require the modules 'seaborn' and 'pandas'")
#
#        sns.set()
#        df = pd.DataFrame()
#        df["Iterations"] = range(len(self.training_loss))
#        df["Training loss"] = self.training_loss_
#        f = sns.lmplot('Iterations', 'Training loss', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5}, fit_reg=False)
#        f.set(yscale = "log")
#
#        if is_none(filename):
#            plt.show()
#        elif is_string(filename):
#            plt.save(filename)
#        else:
#            raise InputError("Wrong data type of variable 'filename'. Expected string")


    def _make_session(self):
        # Force tensorflow to only use 1 thread
        if self.single_thread:
            session_conf = tf.ConfigProto(
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)

            self.session = tf.Session(config = session_conf)
        else:
            self.session = tf.Session()

    def _score(self, *args):
        if self.scoring_function == 'mae':
            return self._score_mae(*args)
        if self.scoring_function == 'rmse':
            return self._score_rmse(*args)
        if self.scoring_function == 'r2':
            return self._score_r2(*args)

    def predict(self, x):
        """
        Use the trained network to make predictions on the data x.

        :param x: The input data of shape (n_samples, n_features)
        :type x: array

        :return: Predictions for the target values corresponding to the samples contained in x.
        :rtype: array

        """

        if is_none(self.session):
            raise InputError("Model needs to be fit before predictions can be made.")

        check_array(x, warn_on_dtype = True)

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/x:0")
            model = graph.get_tensor_by_name("model/model:0")
            y_pred = self.session.run(model, feed_dict = {tf_x : x})
        return y_pred.ravel()

    def fit(self, x, y):
        """
        Fit the neural network to input x and target y.

        :param x: Input data 
        :type x: array of size (n_samples, n_features)
        :param y: Target values
        :type y: array of size (n_samples, )
        :param cost: Computational cost of each feature
        :type cost: array of size (n_samples, n_main_features)

        """
        # Clears the current graph (Makes predictions a bit easier)
        tf.reset_default_graph()

        # Check that X and y have correct shape
        x, y = check_X_y(x, y, multi_output = False, y_numeric = True, warn_on_dtype = True)

        # reshape to tensorflow friendly shape
        y = np.atleast_2d(y).T

        # Collect size input
        self.n_features = x.shape[1]
        self.n_samples = x.shape[0]

        # set n_main_features if previously set to -1
        if self.n_main_features == -1:
            self.n_main_features = self.n_features

        # Set cost to be constant if not passed
        if is_none(self.cost):
            self.cost = np.ones(self.n_main_features)
        elif self.cost.ndim != 1 and self.cost.shape[0] != self.n_features:
            raise InputError("Expected variable 'cost' to have shape (%d, ). Got %s" 
                    % (self.n_features, str(self.cost.shape)))

        # Initial set up of the NN
        with tf.name_scope("Data"):
            tf_x = tf.placeholder(tf.float32, [None, self.n_features], name="x")
            tf_y = tf.placeholder(tf.float32, [None, 1], name="y")

        # Generate weights and biases
        weights, biases = self._generate_weights()
        # Create histogram of weights with tensorboard
        self.tensorboard_logger.write_histogram(weights, biases)

        # Create the graph
        y_pred = self._model(tf_x, weights, biases)

        # Create loss function
        loss = self._loss(y_pred, tf_y, weights)
        # Create summary of loss with tensorboard
        self.tensorboard_logger.write_scalar_summary('loss', loss)

        optimiser = self.optimiser(learning_rate=self.learning_rate).minimize(loss)

        # Initialisation of the variables
        init = tf.global_variables_initializer()
        self.tensorboard_logger.initialise()

        # Create the session
        self._make_session()

        # Running the graph
        self.tensorboard_logger.set_summary_writer(self.session)
        self.session.run(init)

        for i in range(self.iterations):
            feed_dict = {tf_x: x, tf_y: y}
            opt = self.session.run(optimiser, feed_dict=feed_dict)
            self.tensorboard_logger.write_summary(self.session, feed_dict, i, 0)

        # Store the final portfolio weights
        # TODO enable
        self._set_portfolio()

    # TODO this assumes that we actually construct a portfolio
    def _set_portfolio(self):
        self.portfolio = self.portfolio_weights.eval(session = self.session).flatten()

    def _model(self, x, weights, biases = None):
        """
        Constructs the actual network.

        :param x_main: Main input (e.g. method energies)
        :type x_main: tf.placeholder of shape (None, n_main_features)
        :param x_sec: Secondary input (e.g. reaction classes / system charge / multiplicity)
        :type x_sec: tf.placeholder of shape (None, n_features - n_main_features)
        :param weights: Weights used in the network.
        :type weights: list of tf.Variables
        :param biases: Biases used in the network.
        :type weights: list of biases
        :return: Output
        :rtype: tf.Variable of size (None, n_targets)
        """

        with tf.name_scope("model"):
            # indices to keep track of the weights and biases
            # since the various options obscures this
            w_idx, b_idx = 0, 0

            # Make the biases input
            if self.bias_input:
                with tf.name_scope("bias_input"):
                    # Get the main feature slice
                    x_main = x[:,:self.n_main_features]
                    # get the bias
                    b = tf.matmul(x_main, weights[w_idx], name = "input_bias")
                    w_idx += 1
                    # subtract the bias from the main features
                    x_main = tf.subtract(x_main, b, name = "x_main_biased")
                    inp = tf.concat([x_main, x[:,self.n_main_features:]], axis = 1, name = "x_biased")
            else:
                inp = x

            if self.nhl == 0:
                if self.multiplication_layer:
                    h = tf.matmul(inp, weights[w_idx]) + biases[b_idx]
                    b_idx += 1
                    w_idx += 1
                    if self.softmax:
                        h = tf.nn.softmax(h)

                    self.portfolio_weights = h

                    z = tf.reduce_sum(x1 * h, axis = 1, name = "model")
                else:
                    with tf.name_scope("Portfolio_dot_product"):
                        if self.softmax:
                            w = tf.nn.softmax(weights[w_idx], axis = 0, name = "softmax")
                        else:
                            w = weights[w_idx]
                    z = tf.matmul(inp, w, name = "model")
                    self.portfolio_weights = w
                    w_idx += 1
            else:
                if self.nhl >= 1:
                    h = self.activation_function(tf.matmul(inp, weights[w_idx]) + biases[b_idx])
                    b_idx += 1
                    w_idx += 1
                if self.nhl >= 2:
                    h = self.activation_function(tf.matmul(h, weights[w_idx]) + biases[b_idx])
                    b_idx += 1
                    w_idx += 1
                if self.nhl >= 3:
                    h = self.activation_function(tf.matmul(h, weights[w_idx]) + biases[b_idx])
                    b_idx += 1
                    w_idx += 1

                if self.multiplication_layer:
                    h = tf.matmul(h, weights[w_idx]) + biases[b_idx]
                    b_idx += 1
                    w_idx += 1
                    if self.softmax:
                        h = tf.nn.softmax(h)

                    self.portfolio_weights = h
                    z = tf.reduce_sum(x1 * h, axis = 1, name = "model")
                else:
                    z = tf.matmul(h, weights[w_idx], name = "model")
                    self.portfolio_weights = weights[w_idx]
                    w_idx += 1

            if self.fit_bias:
                z += biases[b_idx]

        return z

    def _generate_weights(self):
        """
        Generates the weights.

        :return: tuple of weights and biases
        :rtype: tuple

        """

        weights = []
        biases = []

        with tf.name_scope("weights"):
            # Add a layer that basically calculates a weighted mean.
            # Since some of the methods might be very bad, 
            # this makes more sense than just using the mean
            if self.bias_input:
                weights.append(self._init_weight(self.n_main_features, 1, equal = True, name = "input_bias_weights"))

            # Make the remaining weights in the network
            if self.nhl == 0:
                if self.multiplication_layer:
                    weights.append(self._init_weight(self.n_features,self.n_main_features, name = "multiplication_layer_weights"))
                    biases.append(self._init_bias(self.n_main_features), name = "multiplication_layer_biases")
                else:
                    weights.append(self._init_weight(self.n_features, 1, equal = True, name = "weights_out"))
            else:
                if self.nhl >= 1:
                    weights.append(self._init_weight(self.n_features, self.hl1, name = "weights_in_hl1"))
                    biases.append(self._init_bias(self.hl1))
                if self.nhl >= 2:
                    weights.append(self._init_weight(self.hl1, self.hl2))
                    biases.append(self._init_bias(self.hl2))
                if self.nhl >= 3:
                    weights.append(self._init_weight(self.hl2, self.hl3))
                    biases.append(self._init_bias(self.hl3))

                if self.multiplication_layer:
                    weights.append(self._init_weight(weights[-1].shape[1],self.n_main_features))
                    biases.append(self._init_bias(self.n_main_features))
                else:
                    weights.append(self._init_weight(weights[-1].shape[1],1))


            if self.fit_bias:
                biases.append(self._init_bias(1))

        return weights, biases

    def _loss(self, y_pred, y, weights):
        """
        Constructs the loss function

        :param y_pred: Predicted output
        :type y_pred: tf.Variable of size (None, 1)
        :param y: True output
        :type y: tf.placeholder of shape (None, 1)
        :param weights: Weights used in the network.
        :type weights: list of tf.Variable
        :return: loss
        :rtype: tf.Variable of size (1,)
        """

        with tf.name_scope("loss"):
            with tf.name_scope("l2_loss"):
                loss = tf.nn.l2_loss(y-y_pred)
            if self.l2_reg > 0:
                l2_reg = self._l2_loss(weights)
                loss += l2_reg
            if self.cost_reg > 0:
                # TODO make this general
                cost_reg = self._cost_loss()
                loss += cost_reg

        return loss

    def _init_weight(self, n1, n2, equal = False, name = None):
        """
        Generate a tensor of weights of size (n1, n2)

        """

        if equal:
            w = tf.Variable(np.ones((n1,n2), dtype=np.float32) / (n1 * n2), name = name)
        else:
            w = tf.Variable(tf.truncated_normal([n1,n2], stddev = 1.0 / np.sqrt(n2)), name = name)

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

    def _score_rmse(self, x, y, sample_weight = None):
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

class SingleMethod(BaseModel):
    """
    Selects the single best method.
    """

    def __init__(self, loss = "rmsd", **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._set_loss(loss)
        self.idx = None
        self.portfolio = None

    def _set_loss(self, loss):
        if loss in ["mae", "rmsd", "max"]:
            self.loss = loss
        else:
            raise InputError("Got unknown value %s for parameter 'loss'" % str(loss))

    def fit(self, x, y):
        """
        Choose the single best method.
        """

        self.n_features = x.shape[1]
        self.n_samples = x.shape[0]

        if self.loss == "mae":
            acc = np.mean(abs(x - y[:,None]), axis=0)
        elif self.loss == "rmsd":
            acc = np.sqrt(np.mean((x - y[:,None])**2, axis=0))
        elif self.loss == "max":
            acc = np.max(abs(x - y[:,None]), axis=0)

        self.idx = np.argmin(acc)
        self._set_portfolio()

    def _set_portfolio(self):
        self.portfolio = np.zeros(self.n_features)
        self.portfolio[self.idx] = 1

    def predict(self, x):
        return x[:, self.idx]
