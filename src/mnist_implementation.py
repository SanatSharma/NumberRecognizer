""" 
A program that utilizes backpropagation and stochastic gradient descent (for a feedforward neural network) to correctly 
analyze and comprehend a hand-written number taken from Mnist data
Most of the code here is directly taken or highly influenced by the code 
written by Michael Nielson provided here https://github.com/mnielsen/neural-networks-and-deep-learning
"""

import random
import numpy as numpy

class Mnist_Network(object):

    def __init__(self, size):
        # initialize the number of layers in the network
        self.num_layers = len(size)
        self.size = size
        
        # create a list of random biases for the network
        # randn returns a sample from the standard normal distribution
        # The first layer is the input layer and biases are not set for that since biases help in computing values in seqsequent layers
        self.biases = [numpy.random.randn(val, 1) for val in size[1: ]]
        
        # set weights randomly as well
        self.weights = [numpy.random.randn(y, x) for x, y in zip(size)]