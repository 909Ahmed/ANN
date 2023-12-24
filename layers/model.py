import numpy as np
import math
from functions import *
class Model():

    def __init__(self, input_layer, output_layer):
        
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.layers = self.get_layers()



    def forward_pass (self, sample):

        activations = [sample]
        zlist = []

        for layer in self.layers:
            
            weights = [neuron.weights for neuron in layer.layer]
            bias = [neuron.bias for neuron in layer.layer]

            zlist.append(np.add (np.dot(weights, activations[-1].transpose()), bias))
            activations.append(np.array([sigmoid(z) for z in zlist[-1]]))            

        activations[-1] = softmax(activations[-1])

        return activations, zlist


    def backward_pass(self, activations, zlist, logits):
                
        grad_mat = [x - y for x, y in zip(activations[-1], logits)]
        der_Z = [der_sigmoid(x) for x in zlist[-1]]

        del_curr = [x * y for x, y in zip(grad_mat, der_Z)]        

        del_w_list = []
        del_b_list = []

        for layer in self.layers[::-1]:
            
            prev_act = np.array(activations[-1])
            activations.pop()
            zlist.pop()

            del_weights = np.outer(del_curr, prev_act)
            del_bias = del_curr

            del_w_list.append(del_weights)
            del_b_list.append(del_bias)

            del_curr = self.calc_delta(layer, del_curr, zlist)
        
        return del_w_list, del_b_list
    

    def fit (self, X, Y, epochs, batch_size):
        
        for _ in range(epochs):
            
            count = 0
            
            for i in range(len(X) // batch_size - 1):
                
                del_w, del_b = []
                
                for x, y in zip(X[batch_size * i: batch_size * (i+1)], Y[batch_size * i: batch_size * (i+1)]):

                    activations, zlist = self.forward_pass(x[i])
                    del_wb, del_bb = self.backward_pass(activations, zlist, self.get_embed(x[i]))           
                    count += self.calc_acc (activations[-1], Y[i])

                    del_w += del_wb
                    del_b += del_bb
                
                self.update_weigths(del_w)
                self.update_bias(del_b)

            print (count / len(X))


    def calc_acc (self, predicted, Y):
        return int(predicted.index(max(predicted)) == Y)


    def get_embed (self, logits):
        temp = [0] * 10
        temp[logits] = 1
        return temp

    def get_layers (self):
        
        layers = []
        curr_layer = self.output_layer
        
        while curr_layer != self.input_layer:

            layers.append(curr_layer)
            curr_layer = curr_layer.pre_layer
        
        return layers[::-1]
    

    def calc_delta(self, curr_layer, delta_post, zlist):
        
        if curr_layer.pre_layer == self.input_layer:
            return []
        
        mat = [neuron.weights for neuron in curr_layer.layer]
        mat = np.array(mat)
        mat = mat.transpose()
        
        der_Z = [der_sigmoid(x) for x in zlist[-1]]

        dot = np.dot(mat, delta_post)
        del_curr = [x * y for x, y in zip(dot, der_Z)]

        return del_curr