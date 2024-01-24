import numpy as np
from .mathfunc import *
def cost_der (activations, logits):

    return np.subtract(activations, logits)

def calc_acc (predicted, Y):

    ans = 0
    for pred, y in zip(predicted, Y):
        
        pred = list(pred)
        ans += int(pred.index(max(pred)) == y)
    
    return ans

def get_embed (logits, size):

    temp = [[0] * size] * len(logits)
    
    for index in range(len(logits)):
        temp[index][logits[index]] = 1
            
    return temp

def calc_delta(curr_layer, delta_post, zs):
        
    mat = [neuron.weights for neuron in curr_layer]
    mat = np.array(mat)
    mat = mat.transpose()
    
    der_Z = [[der_sigmoid(x) for x in zs_ele] for zs_ele in zs]

    dot = [np.dot(mat, delta_post_ele) for delta_post_ele in delta_post]
    del_curr = np.multiply(dot, der_Z)

    return del_curr

def drop_func(layer_output, rate):
    
    zeroes = int(rate * len(layer_output))
    ones = len(layer_output) - zeroes

    mult = np.array(np.concatenate((np.zeros(zeroes), np.ones(ones)), axis=0))
    np.random.shuffle(mult)

    return np.multiply(mult, layer_output)