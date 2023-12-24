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

def cost_der (activations, logits):

    return np.subtract(activations, logits)

def calc_acc (predicted, Y):

    predicted = list(predicted)
    return int(predicted.index(max(predicted)) == Y)

def get_embed (logits, size):
    temp = [0] * size
    temp[logits] = 1
    return temp

def calc_delta(curr_layer, delta_post, zs):
        
    mat = [neuron.weights for neuron in curr_layer]
    mat = np.array(mat)
    mat = mat.transpose()
    
    der_Z = [der_sigmoid(x) for x in zs]

    dot = np.dot(mat, delta_post)
    del_curr = np.multiply(dot, der_Z)

    return del_curr
