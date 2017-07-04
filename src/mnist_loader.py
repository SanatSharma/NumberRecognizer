""" Mnist_loader 
A library to load mnist image data.
"""

import cPickle
import gzip

import numpy as numpy

def load_data():
    # training_data is returned and is a tuple with 2 entries
    # The first entry is the training images, which is a numpy ndarray
    # with 50000 entries. Each entry is 28*28 = 784 pixels in every image

    # the second entry is another ndarray containing 50000 entries, which are
    # digit values from 0-9 stating the correct answer to the first entry image

    # training_data is modified to make it easy to use in the neural net
    # using the load_wrapper

    # open gzip
    file = gzip.open('../data/mnist.pkl.gz', 'rb')

    # expand the pickle
    # validation_data and test_data are tuples like training_data or 10000 elements
    training_data, validation_data, test_data = cPickle.load(file)

    file.close()
    return (training_data, validation_data, test_data)

def load_wrapper():
    train, val, test = load_data()

    # load training data
    training_inputs = [numpy.reshape(x, (784,1)) for x in train[0]]
    training_results = [value_result(y) for y in train[1]]
    training_data = zip(training_inputs, training_results)

    # load validation data
    validation_inputs = [numpy.reshape(x, (784, 1)) for x in val[0]]
    validation_data = zip(validation_inputs, val[1])

    # laod test data    
    test_inputs = [numpy.reshape(x, (784, 1)) for x in test[0]]
    test_data = zip(test_inputs, test[1])
    # return tuple containing relevant info
    return (training_data, validation_data, test_data)

def value_result(pos):
    # returns the unit vector with 1 in the pos position denoting
    # the answer and zeros elsewhere
    unit_vect = numpy.zeros((10,1))
    unit_vect[pos] = 1.0
    return unit_vect
    