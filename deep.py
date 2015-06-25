from abc import ABCMeta, abstractmethod

import numpy as np

import theano
import theano.tensor as T

from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer
from lasagne.updates import adagrad, nesterov_momentum
from lasagne.nonlinearities import softmax

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

            epoch = train_history[-1]['epoch']
            new_value = float32(self.ls[epoch-1])
            getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

class EarlyStoppingWindow(object):
    def __init__(self, window=15, patience=10):
        self.window = window
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.errors = []

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        errors.append(current_valid)

        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
            return

        window = self.window
        if current_epoch < 2*window or current_epoch < self.patience:
            return
        
        latest = errors[-window:]
        prev = errors[-2*window:-window]
        if np.mean(latest) > np.mean(prev):
            print("Early stopping due to upward error trend")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

class LayerGenerator(object):
    """
    Abstract method to generate the structure of a neural network
    """
    __metaclass__ = ABCMeta

    def __init__(self, shape):
        self.shape = shape
     
    @abstractmethod
    def input_layer(self, n_units):
        pass

    @abstractmethod
    def hidden_layer(self, n_units):
        pass

    @abstractmethod
    def output_layer(self, n_units):
        pass

    def __iter__(self):
        for layer, n_units in enumerate(shape):
            if layer == 0:
                yield self.input_layer(n_units)
            elif layer == self.n_layers-1:
                yield self.output_layer(n_units)
            else:
                yield self.hidden_layer(n_units)

class DropoutNetworkGenerator(LayerGenerator):
    """
    Simple deep network with dropout
    """
    def __init__(self, shape, corruption, nonlinearity):
        pass

    def input_layer(self):
        return ((InputLayer, {'num_units': self.n_inputs}),)
    
    def hidden_layer(self):
        return (DenseLayer, {'num_units': self.n_units}), \
            (DropoutLayer, {'num_units': self.n_units, 'p': self.corruption})

    def output_layer(self):
        return ((DenseLayer, {'num_units' : self.n_units,  'nonlinearity': self.nonlinearity}),)


class VanillaNet(object):
    """ simple wrapper around the nolearn interface """
    def __init__(self, n_hidden_layers=3, n_hidden_units=100, max_epoch=100):
        self.n__hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.max_epoch = max_epoch

    def fit(self, X, y):
        n_feats = X.shape[1]

