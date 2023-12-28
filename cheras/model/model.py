import numpy as np
from ..Func.mathfunc import *
from ..Func.misc import *
from ..Func.adam import Adam

class Model():

    def __init__(self, input_layer, output_layer):
        
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.layers = self.get_layers()
        adam = Adam(1, 0.9, 0.99, self.layers)

    def forward_pass (self, sample):

        activations = [sample]
        zlist = []

        for layer in self.layers:
            
            weights = [neuron.weights for neuron in layer.layer]
            bias = [neuron.bias for neuron in layer.layer]

            zlist.append(np.add (np.dot(weights, activations[-1]), bias))
            activations.append([sigmoid(z) for z in zlist[-1]])  

        return activations, zlist


    def backward_pass(self, activations, zlist, logits):
                
        grad_mat = cost_der(activations[-1], logits)
        der_Z = [der_sigmoid(x) for x in zlist[-1]]

        del_curr = np.multiply(grad_mat, der_Z)
        del_w_list = []
        del_b_list = []

        for index, layer in enumerate(self.layers[::-1]):
            
            prev_act = activations[-index - 2]
            
            del_weights = np.outer(del_curr, prev_act)
            del_bias = np.array(del_curr)
            
            del_w_list.append(del_weights)
            del_b_list.append(del_bias)

            if (index < len(self.layers) - 1):
                del_curr = calc_delta(layer.layer, del_curr, zlist[-index - 2])

        return del_w_list, del_b_list
    

    def fit (self, X, Y, X_test, Y_test, epochs, batch_size):
        
        for _ in range(epochs):
                        
            for i in range(len(X) // batch_size - 1):
                
                del_w = [np.zeros((layer.size, layer.pre_layer.size)) for layer in self.layers[::-1]]
                del_b = [np.zeros(layer.size) for layer in self.layers[::-1]]
                
                for x, y in zip(X[batch_size * i: batch_size * (i+1)], Y[batch_size * i: batch_size * (i+1)]):

                    activations, zlist = self.forward_pass(x)
                    del_wb, del_bb = self.backward_pass(activations, zlist, get_embed(y, self.output_layer.size))
                    
                    del_w = [np.add(matA, matB) for matA, matB in zip(del_w, del_wb)]
                    del_b = [np.add(listA, listB) for listA, listB in zip(del_b, del_bb)]

                del_w = [matA / batch_size for matA in del_w]
                del_b = [listA / batch_size for listA in del_b]
                
                self.update_weights(del_w)
                self.update_bias(del_b)

            self.evaluate (X_test, Y_test)



    def evaluate (self, X_test, Y_test):
        
        count = 0

        for x, y in zip(X_test, Y_test):
            pred, temp = self.forward_pass(x)
            count += calc_acc(pred[-1], y)
        
        print ((count / len(X_test)) * 100)


    def update_weights(self, del_w):

        new_weights = self.adam.adam_weights (del_w, self.beta1, self.beta2)
        
        for layer, d_we in zip(self.layers[::-1], del_w):
            
            d_we = np.array(d_we)
            d_we = self.adam.lr * d_we
            for i, neuron in enumerate(layer.layer):
                neuron.weights = np.subtract(neuron.weights, d_we[i])
            

    def update_bias(self, del_b):

        new_bias = self.adam.adam_bias (del_b, self.beta1, self.beta2)
            
        for layer, d_b in zip(self.layers[::-1], del_b):
            
            d_b = np.array(d_b)
            d_b = d_b * self.adam.lr
            for i, neuron in enumerate(layer.layer):
                neuron.bias = neuron.bias - d_b[i]


    def get_layers (self):
        
        layers = []
        curr_layer = self.output_layer
        
        while curr_layer != self.input_layer:

            layers.append(curr_layer)
            curr_layer = curr_layer.pre_layer
        
        return layers[::-1]