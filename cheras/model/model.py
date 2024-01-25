import numpy as np
from ..Func.mathfunc import *
from ..Func.misc import *
from ..optimizers.adam import Adam

class Model():

    def __init__(self, input_layer, output_layer):
        
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.layers = self.get_layers()
        self.optimizer = None


    def forward_pass (self, batch, Training=False):

        activations = [batch]
        zlist = []

        for layer in self.layers:
            
            if layer.type == 'dense':

                weights = [neuron.weights for neuron in layer.layer]
                bias = [neuron.bias for neuron in layer.layer]
                
                pre_activations = []
                pre_zlist = []

                for i in range(len(batch)):

                    pre_zlist.append(np.add (np.dot(weights, activations[-1][i]), bias))
                    pre_activations.append([sigmoid(z) for z in pre_zlist[-1]])

                    if layer.drop and Training:
                        pre_zlist[-1] = drop_func(pre_zlist[-1], layer.drop)
                        pre_activations[-1] = drop_func(pre_activations[-1], layer.drop)

                zlist.append(pre_zlist)
                activations.append(pre_activations)

            elif layer.type == 'conv':
                
                # Implementing Forward Pass for Convolutional Layers
                zlist.append(np.add (np.dot(weights, activations[-2]), bias))
                activations.append([sigmoid(z) for z in zlist[-1]])

        return activations, zlist


    def backward_pass(self, activations, zlist, logits):
                
        grad_mat = [cost_der(act, logit) for act, logit in zip(activations[-1], logits)]
        der_Z = [[der_sigmoid(x) for x in pre_zlist] for pre_zlist in zlist[-1]] 

        del_curr = np.multiply(grad_mat, der_Z)
        del_w_list = []
        del_b_list = []

        for index, layer in enumerate(self.layers[::-1]):

            prev_act = activations[-index - 2]
            
            del_weights = [np.outer(del_curr_ele, prev_act_ele) for del_curr_ele, prev_act_ele in zip(del_curr, prev_act)]
            del_bias = np.array(del_curr)
            
            del_w_list.append(del_weights)
            del_b_list.append(del_bias)

            if (index < len(self.layers) - 1):
                del_curr = calc_delta(layer.layer, del_curr, zlist[-index - 2])

        return del_w_list, del_b_list
    

    def fit (self, X, Y, X_valid, Y_valid, epochs, batch_size):
        
        for epoch in range(epochs):
                        
            for batch in range(len(X) // batch_size - 1):
                
                x = X[batch_size * batch: batch_size * (batch+1)]
                y = Y[batch_size * batch: batch_size * (batch+1)]
                y = list(y)

                activations, zlist = self.forward_pass(x, True)
                
                del_wb, del_bb = self.backward_pass(activations, zlist, get_embed(y, self.output_layer.size))
                
                del_wb = [np.sum(del_wbb, axis=0) for del_wbb in del_wb]
                del_bb = [np.sum(del_bbb, axis=0) for del_bbb in del_bb]

                del_wb = [arr / batch_size for arr in del_wb]
                del_bb = [arr / batch_size for arr in del_bb]

                self.update_weights(del_wb, batch_size * epoch + batch)
                self.update_bias(del_bb, batch_size * epoch + batch)

            self.evaluate (X_valid, Y_valid)


    def update_weights(self, del_w, batch):

        new_weights = self.optimizer.adam_weights (del_w, batch)
    
        ign = len(self.layers) - 1

        for index in range(len(new_weights)):
            
            while self.layers[ign - index].type == 'dropout':
                ign-=1
            layer = self.layers[ign - index]
    
            for i, neuron in enumerate(layer.layer):
                neuron.weights = np.subtract(neuron.weights, new_weights[index][i])
            

    def update_bias(self, del_b, batch):

        new_bias = self.optimizer.adam_bias (del_b, batch)
        
        ign = len(self.layers) - 1
        
        for index in range(len(new_bias)):

            while self.layers[ign - index].type == 'dropout':
                ign-=1
            layer = self.layers[ign - index]
            
            for i, neuron in enumerate(layer.layer):
                neuron.bias = neuron.bias - new_bias[index][i]


    def evaluate (self, X_valid, Y_valid):
    
        count = 0
        batch_size = 32 #hard_coder bolte
        for batch in range(len(X_valid) // batch_size - 1):
                
            x = X_valid[batch_size * batch: batch_size * (batch+1)]
            y = Y_valid[batch_size * batch: batch_size * (batch+1)]
            y = list(y)
            
            pred, temp = self.forward_pass(x)
            count += calc_acc(pred[-1], y)
        
        print ((count / len(X_valid)) * 100)


    def optimize (self, lr, beta1, beta2):
        self.optimizer = Adam(self.layers, lr, beta1, beta2)

    def get_layers (self):
        
        layers = []
        curr_layer = self.output_layer
        
        while curr_layer != self.input_layer:

            layers.append(curr_layer)
            curr_layer = curr_layer.pre_layer
        
        return layers[::-1]
    