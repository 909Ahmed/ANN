from sigmoid import sigmoid
import numpy as np

class Neurons(): 

    def __init__(self, weights, bias, activation_fn, values):

        self.weights = np.random.rand(weights)
        self.bias = np.random.random()
        self.activation_fn = activation_fn
        self.value = 0


    
    def calc(self, weights, bias, activation_fn, values):

        activation = sum([a*b for a,b in zip(weights, values)]) + bias
        return activation
    
    def update (self, activations):
        self.value = self.calc(self.weights, self.bias, self.activation_fn, activations)
        return self.value