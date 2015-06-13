import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from activations import squared_error
import time
from progressbar import ProgressBar, Bar, FormatLabel, Timer
from colorama import Fore

def combined_dataset(X,y):
    data = np.c_[y,X]
    return data

def sgd_optimization(X, y, classifier, n_epochs=10, mini_batch_size=600, test_size=0.2, indmat=False, filename=None):
    """ Stochastic gradient descent with early stopping """

    # split into training and testing set
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size)
    
    # predicitons should be indicator matrix form
    lb = None
    if indmat:
        lb = LabelBinarizer()
        lb.fit(train_y)

    # flatten the training and testing data
    train = combined_dataset(train_x, train_y)
    test  = combined_dataset(test_x, test_y)
    
    # parameters
    n_samples = train.shape[0]
    n_train_batches = int(np.floor(n_samples / mini_batch_size))
    patience = 5000 # train over at least this many mini-batches 
    patience_increase = 2 # number of mini-batches to increase if we do better
    validation_frequency = min(n_train_batches, patience / 2.)
    improvement_threshold = 0.995
    best_val_error = np.inf
    end_training = False
    
    # loop through mini-batches
    start_time = time.clock()
    for epoch in xrange(n_epochs):
        np.random.shuffle(train)
        mini_batches = [
            (train[k:k+mini_batch_size, 0], train[k:k+mini_batch_size,1:])
            for k in xrange(0, n_samples, mini_batch_size)]
        
        cost = np.zeros(len(mini_batches))
        for i, mini_batch in enumerate(mini_batches):
            if lb: # transform to indicator matrix
                y_ = lb.transform(mini_batch[0]) 
                cost[i] = classifier.fit(mini_batch[1], y_)
            else:
                cost[i] = classifier.fit(mini_batch[1], mini_batch[0])
            
            # see if we want to check the validation error (NB: i starts at 0)
            iteration = epoch*n_train_batches + i
            if (iteration + 1) % validation_frequency == 0:
                avg_cost = np.mean(cost)
                val_error = 1. - np.mean(classifier.predict(test[:,1:]) == test[:,0])
                print 'epoch %i, avg. cost %f, validation error %f %%'%(
                            epoch,
                            avg_cost,
                            val_error*100
                        )
                if filename:
                    outfile = filename+'_epoch%i_err%.2f.pkl'%(epoch, val_error*100)
                    classifier.save(outfile)
                if val_error < best_val_error:
                    if val_error < best_val_error*improvement_threshold:
                        patience = max(patience, iteration*patience_increase)
                    best_val_error = val_error
        
            if patience < iteration:
                end_training = True
                break

        if end_training:
            end_time = time.clock()
            val_error = 1. - np.mean(classifier.predict(test[:,1:]) == test[:,0])
            print 'Done training: epoch %i, validation error %f %%'%(
                        epoch,
                        val_error*100
                    )
            filename = 'final_err%.2f.pkl'%(val_error*100)
            classifier.save(filename)
            break

def unsupervised_sgd(data, classifier, n_epochs=10, mini_batch_size=600, test_size=0.2, filename=None, verbose=False):
    # split data into training and testing set
    n_samples = data.shape[0]
    proportion = 1.0 - test_size
    n_train_samples = int(n_samples*proportion)
    train, test = np.split(data, [n_train_samples])
    
    # parameters
    n_samples = train.shape[0]
    n_train_batches = int(np.floor(n_samples / mini_batch_size))
    val_frequency = 10
    
    widgets = [FormatLabel(Fore.GREEN+'Epoch: %(value)d/%(max)d '),  Bar(fill='-'), ' ', Timer(), Fore.RESET]
    pbar = ProgressBar(widgets=widgets, maxval=n_epochs)
    for epoch in pbar(xrange(n_epochs)):
        np.random.shuffle(train)
        mini_batches = [train[k:k+mini_batch_size,:] for k in xrange(0, n_samples, mini_batch_size)]

        cost = np.zeros(len(mini_batches))
        for i, mini_batch in enumerate(mini_batches):
            cost[i] = classifier.fit(mini_batch)    
        
        # check average cost / valdation error after every epoch
        if epoch % val_frequency == 0:
            avg_cost = np.mean(cost)
            testhat = classifier.feedforward(test)
            val_error = squared_error(testhat, test)

            if verbose:
                print '\nepoch %i, avg. cost %f, validation error %f'%(
                            epoch,
                            avg_cost,
                            val_error
                        )
            if filename:
                outfile = filename+'_epoch%i_err%.2f.pkl'%(epoch, val_error*100)
                classifier.save(outfile) 
    
    # done training / save output
    testhat = classifier.feedforward(test)
    val_error = squared_error(testhat, test)
    if verbose:
        print '\nDone training: epoch %i, validation error %f'%(
                    epoch,
                    val_error
                )
    if filename:
        outfile = filename+'_final_err%.2f.pkl'%(val_error*100)
        classifier.save(outfile)
