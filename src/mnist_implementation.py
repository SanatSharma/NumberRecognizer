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
        self.weights = [numpy.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]

    # Feedforward returns the output of the network, with value being the input into the network
    def feedforward(self, value):
        for bias, weight in zip(self.biases, self.weights):
            # the feedforward function uses the simple sigmoid(w.x +b) formula
            value = sigmoid(numpy.dot(weight, value) + bias)
        return value
    
    def sigmoid(function_val):
        return 1.0/(1.0 + numpy.exp(-function_val))
    
    # derivative of the sigmoid function
    def sigmoid_derivative(derivative_input):
        return sigmoid(derivative_input) * (1 - sigmoid(derivative_input))

    def train(self, training_data, epochs, mini_batch_size, eta):
        # function to train the neural network using a mini-batch stochastic
        # gradient descent. training data is a list of tuples with training data and 
        # wanted output.
        training_data_length = len(training_data)
        # for every epoch, generate mini-batches and train on them
        for i in xrange(epochs):
            random.shuffle(training_data)
            
            mini_batches = [training_data[j: j+mini_batch_size] 
            for j in xrange(0, training_data_length, mini_batch_size)]

            # update the mini_batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            print "Epoch {0} completed".format(i)

    def update_mini_batch(self, mini_batch, eta):
        # update the weights and biases for the next layer by applying
        # gradient descent using backpropogation to a mini batch

        # initialize
        change_bias = [numpy.zeros(bias.shape) for bias in self.biases]
        change_weight = [numpy.zeros(weight.shape) for weight in self.weights]

        # mini_batch is a list of tuples with x,y values
        for x, y in mini_batch:
            delta_bias, delta_weight = self.backpropagation(x,y)
            change_bias = [nb + db for nb, db in zip(change_bias, delta_bias)]
            change_weight = [nw + dw for nw,dw in zip(change_weight, delta_weight)]

        self.biases = [bias - (eta/len(mini_batch)) * nb
                        for bias, nb in zip(selg.biases, change_bias)]
        self.weights = [weight - (eta/len(mini_batch)) * nw 
                        for weight, nw in zip(self.weights, change_weight)]

    
    def backpropagation(self, x, y):
        # function returns a tuple that contains the gradient for the cost
        # function C_x. The tuple contains the change_bias and change_weight
        # which are layer-by-ayer lists of numpy arrays, like self.biases
        change_bias = [numpy.zeros(bias.shape) for bias in self.biases]
        change_weight = [numpy.zeros(weight.shape) for weight in self.weights]

        """ The following code is taken from Michael Nielson's solution verbose"""
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        change_bias[-1] = delta
        change_weight[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book. Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on. It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            change_bias[-l] = delta
            change_weight[-l] = np.dot(delta, activations[-l-1].transpose())
        return (change_bias, change_weight)
        """ end of verbose copy"""
    

    