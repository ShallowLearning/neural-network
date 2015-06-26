from abc import ABCMeta, abstractmethod

import numpy as np

import theano
import theano.tensor as T

from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer
from lasagne.updates import adagrad, nesterov_momentum, adadelta
from lasagne.nonlinearities import softmax

def float32(k):
    return np.cast['float32'](k)

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
    
    def load_best_weights(self, nn, train_history):
        nn.load_params_from(self.best_weights)

class EarlyStoppingWindow(object):
    def __init__(self, window=15, patience=10):
        self.window = window
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.errors = []

    def __call__(self, nn, train_history):
        window = self.window
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        self.errors.append(current_valid)
        
        # save best parameters
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_weights = nn.get_all_params_values()
            return

        # make sure we train long enough for heuristic
        if current_epoch < 2*window or current_epoch < self.patience:
            return
        
        latest = self.errors[-window:]
        prev = self.errors[-2*window:-window]
        if np.mean(latest) > np.mean(prev):
            print("Early stopping due to upward error trend")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()
    
    def load_best_weights(self, nn, train_history):
        nn.load_params_from(self.best_weights)

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
        n_layers = len(self.shape)
        for layer, n_units in enumerate(self.shape):
            if layer == 0:
                yield self.input_layer(n_units)
            elif layer == n_layers-1:
                yield self.output_layer(n_units)
            else:
                yield self.hidden_layer(n_units)

class DropoutNetworkGenerator(LayerGenerator):
    """
    Simple deep network with dropout
    """
    def __init__(self, shape, corruption=0.5, nonlinearity=None):
        self.shape = shape
        self.corruption = corruption
        self.nonlinearity = nonlinearity

    def input_layer(self, n_units):
        return (InputLayer, {'shape': n_units})
    
    def hidden_layer(self, n_units):
        if self.corruption is None:
            return (DenseLayer, {'num_units': n_units})
        
        if self.corruption < 0 or self.corruption > 1:
            raise ValueError("`corruption must lie in the interval [0,1]")
        else:
            return [(DenseLayer, {'num_units': n_units}), \
                (DropoutLayer, {'p': self.corruption})]

    def output_layer(self, n_units):
        return (DenseLayer, {'num_units' : n_units,  'nonlinearity': self.nonlinearity})

def flatten(lst):
    for l in lst:
        if isinstance(l, list):
            for sub in l:
                yield sub
        else:
            yield l

class UpdateType(object):
    Nesterov = 1
    Adagrad = 2
    AdaDelta = 3

UPDATES = {
            UpdateType.Nesterov : nesterov_momentum,
            UpdateType.Adagrad : adagrad,
            UpdateType.AdaDelta : adadelta
}

class VanillaNet(object):
    """ simple wrapper around the nolearn interface """
    def __init__(self, n_hidden_layers=3, n_hidden_units=100, max_epoch=100, 
        corruption=None, stop_window=15, patience=10, update=UpdateType.Nesterov,
        learning_rate=0.01, momentum=0.9):

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.max_epoch = max_epoch
        self.corruption = corruption
        self.stop_window = stop_window
        self.patience = patience
        self.update = update
        self.learning_rate = learning_rate
        self.momentum = momentum

    
    def structure(self):
        net_generator = DropoutNetworkGenerator(shape=self.shape, \
            corruption=self.corruption, nonlinearity=softmax)
        layers = list(flatten([layer for layer in net_generator]))
        return layers

    def network(self):
        # determine heuristics
        early_stopping = EarlyStoppingWindow(window=self.stop_window, patience=self.patience)
        update = UPDATES[self.update]
        params = dict(layers=self.arch,
                      update=update,
                      update_learning_rate=theano.shared(float32(self.learning_rate)),
                      on_epoch_finished=[
                        AdjustVariable('update_learning_rate', start=self.learning_rate, stop=0.0001),
                        early_stopping
                      ],
                      on_training_finished=[early_stopping.load_best_weights],
                      eval_size=0.2,
                      max_epochs=self.max_epoch,
                      verbose=1,
                      )

        if self.update not in [UpdateType.Adagrad]:
            params['update_momentum'] = theano.shared(float32(self.momentum))
            params['on_epoch_finished'].append(AdjustVariable('update_momentum', \
                start=self.momentum, stop=0.999))

        return NeuralNet(**params)

    def fit(self, X, y):
        n_feats = X.shape[1]
        n_output = len(np.unique(y))

        # determine network structure
        self.shape = [(None, n_feats)] + [self.n_hidden_units for _ in range(self.n_hidden_layers)] + [n_output]
        self.arch = self.structure()
        self.model = self.network()

        # train model
        self.model.fit(X, y)

        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError('The model must be fit first')
        return self.model.predict(X)
