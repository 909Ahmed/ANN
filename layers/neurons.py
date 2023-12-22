from sigmoid import sigmoid

class Neurons(): 

    def __init__(self, weights, bias, activation_fn, values):

        self.weights = [1] * weights
        self.bias = bias
        self.activation_fn = activation_fn
        self.value = 0
        # self.value = self.calc(weights, bias, activation_fn, values)


    
    def calc(self, weights, bias, activation_fn, values):

        activation = sum([a*b for a,b in zip(weights, values)]) + bias
        return activation
    
    def update (self, activations):
        self.value = self.calc(self.weights, self.bias, self.activation_fn, activations)
        return self.value