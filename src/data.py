import gzip
import pickle

import numpy
import theano

from scipy.ndimage import imread
from scipy.misc import imresize

from aux import data_path, script_directory

def loadMNIST(file_path, shape):
    
    with gzip.open(file_path, 'rb') as data:
        training_set, validation_set, test_set = pickle.load(data)
    
    X_train, y_train = training_set
    X_valid, y_valid = validation_set
    X_test, y_test = test_set
    
    X_train = X_train.reshape(-1, *shape)
    X_valid = X_valid.reshape(-1, *shape)
    X_test = X_test.reshape(-1, *shape)
    
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def loadHomemade(file_paths, shape):
    
    N_homemade = len(file_paths)
    
    if len(shape) == 1:
        N = shape[0]
        x_LR_HM = numpy.zeros((N_homemade, N))
    elif len(shape) == 3:
        C, h, w = shape
        x_LR_HM = numpy.zeros((N_homemade, C, h, w))
    
    for i, file_path in enumerate(file_paths):
        
        if ".png" in file_path:
            image = imread(file_path)
            x_LR_HM_example = numpy.around(image.mean(axis = 2) / image.max())
        elif ".txt" in file_path:
            x_LR_HM_example = numpy.loadtxt(file_path)
        
        x_LR_HM[i, :] = x_LR_HM_example.reshape(*shape)
    
    return x_LR_HM

if __name__ == '__main__':
    script_directory()
    
    from super_resolution import bernoullisample
    
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = loadMNIST(data_path("mnist.pkl.gz"), [28**2])
    
    print((X_train - bernoullisample(X_train)).max())
