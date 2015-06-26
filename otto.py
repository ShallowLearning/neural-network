from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.cross_validation import train_test_split

import theano
import theano.tensor as T

import pandas as pd
import numpy as np

from deep import VanillaNet, float32, UpdateType, NonLinearityType

import cPickle as pickle
import sys

sys.setrecursionlimit(10000)

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

def test_data():
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
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = float32(X)

    # get test data
    Xtest = pd.read_csv('test.csv')
    Xtest = Xtest.drop('id', axis=1)
    Xtest = scaler.transform(Xtest)
    Xtest = float32(Xtest)
    
    return Xtest, encoder

if __name__ == '__main__':
    #X, y, shape = generate_data()
    #net = VanillaNet(n_hidden_layers=2,
    #                 n_hidden_units=500,
    #                 max_epoch=100,
    #                 dropout_p=0.5,
    #                 update=UpdateType.Adagrad,
    #                 activation=NonLinearityType.LeakyRectify,
    #                 learning_rate=0.04)
    #net.fit(X, y)
    #pickle.dump(net, open('net.pickle', 'w'))
    
    net = pickle.load(open('otto_net.pickle', 'r'))
    Xtest, encoder = test_data()
    yhat = net.model.predict_proba(Xtest)
    #df = pd.DataFrame({'id':np.arange(1,n_features), 
