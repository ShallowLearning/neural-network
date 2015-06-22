import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.testing as npt
import cPickle as pickle
import theano
import theano.tensor as T
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split

from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, \
    GaussianNoiseLayer, get_output
from lasagne.updates import adagrad, nesterov_momentum
from lasagne.nonlinearities import softmax

def float32(k):
    return np.cast['float32'](k)

def generate_data():
    """ process datafile for Lasagne """
    X = pd.read_csv('train.csv')
    X = X.drop('id', axis=1)

    # shuffle data
    X = X.reindex(np.random.permutation(X.index))

    # extracta and encode target
    y = X.target.values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y).astype(np.int32)
    
    # remove target
    X = X.drop('target', axis=1)

    # standardize the data
    X = StandardScaler().fit_transform(X)
    X = float32(X)

    # data shape
    num_features = X.shape[1]
    num_classes = len(encoder.classes_)

    return np.array(X), y, (num_features, num_classes)


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

def get_all_params_values(nn):
    return_value = OrderedDict()
    for name, layer in nn.layers_.items():
        return_value[name] = [p.get_value() for p in layer.get_params()]
    return return_value

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

#class NoisyBatchIterator(BatchIterator):
#    def __init__(self, corruption, *args, **kwargs):
#        super(NoisyBatchIterator, self).__init__(*args, **kwargs)
#        self.corruption = corruption
#
#    def transform(self, Xb, yb):
#        Xb, yb = super(NoisyBatchIterator, self).transform(Xb, yb)
#
#        # apply a bernoulli mask to the input
#        Xb = np.random.binomial(n=1,p=1-self.corruption,size=Xb.shape) * Xb
#
#        return Xb, yb
#
#def get_dA(shape):
#    layers = [ ('input', InputLayer),
#               ('hidden', DenseLayer),
#               ('output', DenseLayer),
#             ]
#    
#    params = dict(layers=layers,
#                  input_shape=(None, shape[0]),
#                  hidden_num_units=shape[1],
#                  output_num_units=shape[0],
#                  output_nonlinearity=None,
#                  update=nesterov_momentum,      
#                  update_learning_rate=theano.shared(float32(0.03)),
#                  update_momentum=theano.shared(float32(0.9)),
#                  on_epoch_finished=[
#                    AdjustVariable('update_learning_rate', start=0.04, stop=0.0001),
#                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
#                    EarlyStopping(patience=10),
#                  ],
#                  eval_size=0.2,                 
#                  regression=True,
#                  batch_iterator_train=NoisyBatchIterator(batch_size=128, corruption=0.3),
#                  max_epochs=10,
#                  verbose=1,                     
#                  )
#
#    return NeuralNet(**params)

def get_dA(shape):

    layers = [ ('input', InputLayer),
               ('noise', DropoutLayer),
               ('hidden', DenseLayer),
               ('output', DenseLayer),
             ]
    
    params = dict(layers=layers,
                  input_shape=(None, shape[0]),
                  noise_p=0.4,
                  hidden_num_units=shape[1],
                  output_num_units=shape[0],
                  output_nonlinearity=None,
                  update=nesterov_momentum,      
                  update_learning_rate=theano.shared(float32(0.03)),
                  update_momentum=theano.shared(float32(0.9)),
                  on_epoch_finished=[
                    AdjustVariable('update_learning_rate', start=0.04, stop=0.0001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    EarlyStopping(patience=10),
                  ],
                  eval_size=0.2,                 
                  regression=True,
                  max_epochs=1000,
                  verbose=1,                     
                  )

    return NeuralNet(**params)

def get_nnet(shape):
    layers = [ ('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout1', DropoutLayer),
               ('output', DenseLayer) ]
    
    params = dict(layers=layers,
                  input_shape=(None, shape[0]),
                  dense0_num_units=5000, 
                  dropout0_p=0.4,                
                  dense1_num_units=1000,          
                  dropout1_p=0.4,                
                  output_num_units=shape[1],     
                  output_nonlinearity=softmax,   
                  update=nesterov_momentum,      
                  update_learning_rate=theano.shared(float32(0.03)),
                  update_momentum=theano.shared(float32(0.9)),
                  on_epoch_finished=[
                    AdjustVariable('update_learning_rate', start=0.04, stop=0.0001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    EarlyStopping(patience=10),
                  ],
                  eval_size=0.2,                 
                  verbose=1,                     
                  max_epochs=1000
                  )

    return NeuralNet(**params)

def get_pretrained_nnet(shape):
    layers = [ ('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout1', DropoutLayer),
               ('dense2', DenseLayer),
               ('output', DenseLayer) ]
    
    params = dict(layers=layers,
                  input_shape=(None, shape[0]),
                  dense0_num_units=shape[1], 
                  dropout0_p=0.4,                
                  dense1_num_units=shape[2],          
                  dropout1_p=0.4,                
                  dense2_num_units=shape[3],
                  output_num_units=shape[4],     
                  output_nonlinearity=softmax,   
                  update=nesterov_momentum,      
                  update_learning_rate=theano.shared(float32(0.03)),
                  update_momentum=theano.shared(float32(0.9)),
                  on_epoch_finished=[
                    AdjustVariable('update_learning_rate', start=0.04, stop=0.0001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    EarlyStopping(patience=10),
                  ],
                  eval_size=0.2,                 
                  verbose=1,                     
                  max_epochs=100
                  )

    return NeuralNet(**params)

def unsupervised_pretraining(X):
    """ performe unsupervised pre-training """
    Z = X
    names = ['dense0', 'dense1', 'dense2', 'output']
    shapes = [(93,100), (100,50), (50,25), (25,9)]
    params = dict()
    xs = T.matrix('xs').astype(theano.config.floatX)
    for name, shape in zip(names, shapes):
        dA = get_dA(shape)
        dA.fit(Z,Z)
        get_hidden_features = theano.function([xs], get_output(dA.layers_['hidden'], xs, deterministic=True))
        Z = float32(get_hidden_features(Z))
        params[name] = dA.get_all_params_values()['hidden']
     
    # build the actual network
    net = get_pretrained_nnet([93, 100, 50, 25, 9])
    net.load_params_from(params)
    return net
    

if __name__ == '__main__':
    X, y, shape = generate_data()
    
    #net = unsupervised_pretraining(X)
    net = get_nnet(X.shape)
    net.fit(X,y)
