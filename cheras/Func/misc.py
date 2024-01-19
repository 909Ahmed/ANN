import numpy as np
from .mathfunc import *
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

def drop_func(layer_output, rate):
    
    zeroes = int(rate * len(layer_output))
    ones = len(layer_output) - zeroes

    mult = np.array(np.concatenate((np.zeros(zeroes), np.ones(ones)), axis=0))
    np.random.shuffle(mult)

    return np.multiply(mult, layer_output)