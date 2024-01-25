import math
import numpy as np

def sigmoid (x):

    return math.exp(x) / (1 + math.exp(x))
    
def der_sigmoid (x):

    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    
    sigma = sum([math.exp(out) for out in x])
    res = [(math.exp(out) / sigma) for out in x]
    
    return np.array(res)

def relu (x):
    return max (0, x)

def LeakyReLU (x, alpha):
    return max (alpha * x, x)