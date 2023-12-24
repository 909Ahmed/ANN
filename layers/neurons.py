from functions import sigmoid
import numpy as np

class Neurons(): 

    def __init__(self, weights, activation_fn):

        self.weights = np.random.randn(weights)
        self.bias = np.random.rand()