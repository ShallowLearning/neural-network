from abc import ABCMeta, abstractmethod

import numpy as np

import theano
import theano.tensor as T

from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer
from lasagne.updates import sgd, momentum, nesterov_momentum, adagrad, rmsprop, \
    adadelta, adam
from lasagne.nonlinearities import softmax, sigmoid, tanh, rectify, \
    leaky_rectify, very_leaky_rectify, linear, identity

def float32(k):
    """ helper function to make theano happy (numbers need to be float32) """
    return np.cast['float32'](k)


# Update Nonlinearities
#---------------------------------------------------------------------------------------
class NonLinearityType(object):
    """ Enum for nonlinearities """
    Sigmoid = 1
    Softmax = 2
    Tanh = 3
    Rectify = 4
    LeakyRectify = 5
    VeryLeakyRectify = 6
    Linear = 7


NONLINEARITIES = {
                    NonLinearityType.Sigmoid : sigmoid,
                    NonLinearityType.Softmax : softmax,
                    NonLinearityType.Tanh : tanh,
                    NonLinearityType.Rectify : rectify,
                    NonLinearityType.LeakyRectify : leaky_rectify, # alpha = 0.01
                    NonLinearityType.VeryLeakyRectify : very_leaky_rectify, # alpha = 1/3
                    NonLinearityType.Linear : linear
}


# Update rules
#---------------------------------------------------------------------------------------
class UpdateType(object):
    """ Enum for different update algorithms """
    SGD = 1      # stochastic gradient descent
    Momentum = 2 # SGD with momentum
    Nesterov = 3 # SGD with nesterov momentum (prefered over normal momentum)
    Adagrad = 4  # Adagrad
    Rmsprop = 5  # Rmsprop
    AdaDelta = 6 # Adadelta
    Adam = 7     # Adam


# a dictionary mapping between the enums and the lasgane functions
UPDATES = {
            UpdateType.SGD : sgd,
            UpdateType.Momentum : momentum,
            UpdateType.Nesterov : nesterov_momentum,
            UpdateType.Adagrad : adagrad,
            UpdateType.Rmsprop : rmsprop,
            UpdateType.AdaDelta : adadelta,
            UpdateType.Adam : adam,
}


# Variable shedule heuristics
#---------------------------------------------------------------------------------------
class BaseSchedule(object):
    """ Base for a variable update schedule """ 
    __metaclass__ = ABCMeta
    
    def __init__(self, name, start):
        self.name = name
        self.start = start
        self.schedule = None


    @abstractmethod
    def determine_schedule(self, max_epochs):
        """ schedule generator """
        pass


    def __call__(self, nn, train_history):
        # get the shedule
        if self.schedule is None:
            self.schedule = self.determine_schedule(nn.max_epochs)

        # update the parameters
        epoch = train_history[-1]['epoch']
        new_value = float32(self.schedule[epoch-1])
        getattr(nn, self.name).set_value(new_value)


class ConstantSchedule(BaseSchedule):
    """ Simply return the same value """
    def determine_schedule(self, max_epochs):
        return np.repeat(self.start, max_epochs)


class LinearSchedule(BaseSchedule):
    """ Linear update schedule """
    def __init__(self, name, start, stop):
        self.value = start
        self.stop = stop
        self.delta = None
        super(LinearSchedule, self).__init__(name, start)


    def determine_schedule(self, max_epochs):
        if self.delta is None:
            self.delta = (self.stop - self.start) / max_epochs

        return np.repeat(self.value, max_epochs) + np.arange(max_epochs)*self.delta


class ExponentialSchedule(BaseSchedule):
    """ An exponential update schedule """
    def __init__(self, name, start, factor):
        self.value = start
        self.factor = factor
        super(ExponentialSchedule, self).__init__(name, start)


    def determine_schedule(self, max_epochs):
        if np.abs(self.factor) > 1 or np.abs(self.factor) < 0:
            raise ValueError('Exponential schedule only excepts growth factors with \
            magnitudes in the range [0,1]')
        return np.repeat(self.value, max_epochs) * \
            np.power(self.factor, np.arange(max_epochs))
        

# Early stopping heuristics
#---------------------------------------------------------------------------------------
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


# BatchIterator Extensions (good for augmenting features)
#---------------------------------------------------------------------------------------
def shuffle(*arrays):
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]

class ShufflingBatchIteratorMixin(object):
    """ Mixin for shuffling data after each epoch """
    def __iter__(self):
        self.X, self.y = shuffle(self.X, self.y)
        for res in super(ShufflingBatchIteratorMixin, self).__iter__():
            yield res

class ShuffleBatchIterator(ShufflingBatchIteratorMixin, BatchIterator):
    pass

# Network Architecture Generators
#---------------------------------------------------------------------------------------
class NetworkGenerator(object):
    """
    Abstract method to generate the structure of a neural network.
    Could be more general (assumes all same hidden layers, not a great assumption)
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


    def __call__(self):
        n_layers = len(self.shape)
        structure = []
        for layer, n_units in enumerate(self.shape):
            if layer == 0:
                structure.append(self.input_layer(n_units))
            elif layer == n_layers-1:
                structure.append(self.output_layer(n_units))
            else:
                structure.append(self.hidden_layer(n_units))
        
        return structure


class DropoutNetworkGenerator(NetworkGenerator):
    """
    Simple deep network with dropout
    """
    def __init__(self, shape, dropout_p=0.5, activation=rectify, output_nonlinearity=None):
        self.shape = shape
        self.dropout_p = dropout_p
        self.output_nonlinearity = output_nonlinearity
        self.activation = activation


    def input_layer(self, n_units):
        return (InputLayer, {'shape': n_units})


    def hidden_layer(self, n_units):
        if self.dropout_p is None:
            return (DenseLayer, {'num_units': n_units, 'nonlinearity': self.activation})
        
        if self.dropout_p < 0 or self.dropout_p > 1:
            raise ValueError("Dropout probability  must lie in the interval [0,1].")
        else:
            return [
                (DenseLayer, {'num_units': n_units, 'nonlinearity': self.activation}), \
                (DropoutLayer, {'p': self.dropout_p})
                ]


    def output_layer(self, n_units):
        return (DenseLayer, {'num_units' : n_units, \
            'nonlinearity': self.output_nonlinearity})


def flatten(lst):
    """ Generator to unravel the network structure.
        Should be made more general for more obscure architectures.
    """
    for l in lst:
        if isinstance(l, list):
            for sub in l:
                yield sub
        else:
            yield l


# Wrappers around the nolearn + lasagne interface
#---------------------------------------------------------------------------------------
class BaseNet(object):
    """ simple wrapper around the nolearn interface """
    __metaclass__ = ABCMeta
    network = None
    network_params = None
    
    def __init__(self, n_hidden_layers=3, n_hidden_units=100, max_epoch=100, 
        dropout_p=None, stop_window=15, patience=10, update=UpdateType.Nesterov,
        learning_rate=0.01, momentum=0.9, output_nonlinearity=NonLinearityType.Softmax,
        activation=NonLinearityType.Rectify):

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.max_epoch = max_epoch
        self.dropout_p = dropout_p
        self.stop_window = stop_window
        self.patience = patience
        self.update = update
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activation = activation
        self.output_nonlinearity = output_nonlinearity
        self.shape = None
    
    @property
    @abstractmethod
    def network_parameters(self):
        pass


    def _get_network_generator(self):
        return self.network(**self.network_parameters)


    def structure(self):
        net_generator = self._get_network_generator()
        layers = list(flatten(net_generator()))
        return layers


    def build_network(self):
        # determine heuristics
        early_stopping = EarlyStoppingWindow(window=self.stop_window, \
            patience=self.patience)
        update = UPDATES[self.update]
        params = dict(layers=self.arch,
                      update=update,
                      update_learning_rate=theano.shared(float32(self.learning_rate)),
                      on_epoch_finished=[
                        LinearSchedule('update_learning_rate', start=self.learning_rate,\
                            stop=0.0001),
                        early_stopping
                      ],
                      batch_iterator_train=ShuffleBatchIterator(batch_size=128),
                      eval_size=0.2,
                      max_epochs=self.max_epoch,
                      verbose=1,
                      )

        if self.update not in [UpdateType.Adagrad]:
            params['update_momentum'] = theano.shared(float32(self.momentum))
            params['on_epoch_finished'].append(LinearSchedule('update_momentum', \
                start=self.momentum, stop=0.999))

        return NeuralNet(**params)


    def fit(self, X, y):
        n_feats = X.shape[1]
        n_output = len(np.unique(y))

        # determine network structure
        if self.shape is None:
            self.shape = [(None, n_feats)] + \
                [self.n_hidden_units for _ in range(self.n_hidden_layers)] + [n_output]
        self.arch = self.structure()
        self.model = self.build_network()

        # train model
        self.model.fit(X, y)

        return self

    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError('The model must be fit first')
        return self.model.predict(X)


    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError('The model must be fit first')
        return self.model.predict_proba(X)


# Define your networks here
#---------------------------------------------------------------------------------------
class VanillaNet(BaseNet):
    network = DropoutNetworkGenerator
    
    @property
    def network_parameters(self):
        if self.network_params is None:
            self.network_params = {
               'shape': self.shape, 
               'dropout_p': self.dropout_p,
               'activation': NONLINEARITIES[self.activation],
               'output_nonlinearity': NONLINEARITIES[self.output_nonlinearity]
        }
        return self.network_params 
