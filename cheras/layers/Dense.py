from .neurons import Neurons
class Dense ():

    def __init__(self, size, activation_fn, pre_layer):
        
        self.size = size
        self.pre_layer = pre_layer
        self.layer = [Neurons(self.pre_layer.size) for _ in range(size)]
        self.activation_fn = activation_fn
        self.type = 'dense'