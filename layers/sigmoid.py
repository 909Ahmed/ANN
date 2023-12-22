import math

def sigmoid (x):

    return math.exp(x) / (1 + math.exp(x))

def der_sigmoid (x):

    return sigmoid(x) * (1 - sigmoid(x))