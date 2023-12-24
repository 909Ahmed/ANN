import math
import numpy as np
def sigmoid (x):

    return math.exp(x) / (1 + math.exp(x))

def der_sigmoid (x):

    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    
    sum = 0
    res = []
    
    for out in x:
        exp_out = math.exp(out)
        sum += exp_out

    for out in x:
        res.append(math.exp(out) / sum)
    
    return np.array(res)

def relu (x):
    return max (0, x)

def LeakyReLU (x, alpha):
    return max (alpha * x, x)