from neurons import Neurons

class Layer ():

    def __init__(self, size, activation_fn, pre_layer):
        
        self.size = size
        self.pre_layer = pre_layer
        self.layer = [Neurons(self.pre_layer.size, 1, 'sigmoid', self.pre_layer.values) for _ in range(size)]
        self.values = [neuron.value for neuron in self.layer]

    def update (self, activations):

        self.values = [neuron.update(activations) for neuron in self.layer]
        return self.values