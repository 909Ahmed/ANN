import numpy as np
from sigmoid import der_sigmoid

class Model():

    def __init__(self, input_layer, output_layer):
        
        self.input_layer = input_layer,
        self.output_layer = output_layer,
        self.layers = self.get_layers()


    def forward_pass (self, sample):

        pre_layer = self.input_layer
        pre_layer.output = sample

        for layer in self.layers:
            layer.update(pre_layer.output)
            pre_layer = layer

    def backward_pass(self, logits):
        
        curr_layer = self.output_layer

        grad_mat = [x - y for x, y in zip(self.output_layer.output, logits)]
        der_Z = [der_sigmoid(x) for x in self.output_layer.values]

        delta_list = []
        del_curr = [x * y for x, y in zip(grad_mat, der_Z)]
        delta_list.append(del_curr)
        
        while curr_layer != self.input_layer:
            
            delta_post = delta_list[-1]
            
            mat = [neuron.weights for neuron in curr_layer.layer]
            mat = np.array(mat)
            mat = mat.transpose()
            
            der_Z = [der_sigmoid(x) for x in self.curr_layer.values]
            
            dot = np.dot(mat, delta_post)
            del_curr = [x * y for x, y in zip(dot, der_Z)]
            delta_list.append(del_curr)

            curr_layer = curr_layer.pre_layer



    def get_layers (self):
        
        layers = [self.output_layer]
        curr_layer = self.output_layer
        
        while curr_layer != self.input_layer:

            layers.append(curr_layer)
            curr_layer = curr_layer.pre_layer

        # layers.append(curr_layer)
        return layers[::-1]