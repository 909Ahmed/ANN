import math

def sigmoid (activation):

    return math.exp(activation) / (1 + math.exp(activation))