import numpy

import theano

import gzip

from aux import data_path

def load():

    def load_mnist_images(filename):
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = numpy.frombuffer(f.read(), numpy.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 28**2)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / numpy.float32(256)

    def load_mnist_labels(filename):
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = numpy.frombuffer(f.read(), numpy.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images(data_path('train-images-idx3-ubyte.gz'))
    y_train = load_mnist_labels(data_path('train-labels-idx1-ubyte.gz'))
    X_test = load_mnist_images(data_path('t10k-images-idx3-ubyte.gz'))
    y_test = load_mnist_labels(data_path('t10k-labels-idx1-ubyte.gz'))

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    
    def shared_dataset(x, y, borrow=True):
        shared_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return shared_dataset(X_train, y_train), shared_dataset(X_val, y_val), shared_dataset(X_test, y_test)
