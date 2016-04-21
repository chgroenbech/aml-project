import gzip
import pickle

import numpy
import theano

from aux import data_path, script_directory

def load(file_path, shape):
    
    with gzip.open(file_path, 'rb') as data:
        training_set, validation_set, test_set = pickle.load(data)
    
    X_train, y_train = training_set
    X_valid, y_valid = validation_set
    X_test, y_test = test_set
    
    X_train = numpy.around(X_train)
    X_valid = numpy.around(X_valid)
    X_test = numpy.around(X_test)
    
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

if __name__ == '__main__':
    script_directory()
    load(data_path("mnist.pkl.gz"), shape = [1])
