import numpy as np
import math
class Adam ():


    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.99):
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = 0.00000001
        
        self.weight_vel = [np.zeros(layer.size, layer.pre_layer.size) for layer in layers[::-1]]
        self.bias_vel = [np.zeros(layer.size) for layer in layers[::-1]]

        self.weight_rms = [np.zeros(layer.size, layer.pre_layer.size) for layer in layers[::-1]]
        self.bias_rms = [np.zeros(layer.size) for layer in layers[::-1]]

    
    def adam_weights (self, del_w):
        
        temp1 = [self.calc_vel_weights(mat, index) for index, mat in enumerate(del_w)]
        temp2 = [self.calc_rms_weights(mat, index) for index, mat in enumerate(del_w)]

        self.weight_vel = temp1
        self.weight_rms = temp2

        return [vel / math.sqrt (rms + self.eps) for vel, rms in zip(temp1, temp2)]
        

    def adam_bias (self, del_b):
        
        temp1 = [self.calc_vel_bias(mat, index) for index, mat in enumerate(del_b)]
        temp2 = [self.calc_rms_bias(mat, index) for index, mat in enumerate(del_b)]

        self.bias_vel = temp1
        self.bias_rms = temp2

        return [vel / math.sqrt (rms + self.eps) for vel, rms in zip(temp1, temp2)]
    


    def calc_vel_weights (self, mat, index):

        return self.beta1 * self.weight_vel[index] + (1 - self.beta1) * mat

    def calc_rms_weights (self, mat, index):

        return self.beta2 * self.weight_rms[index] + (1 - self.beta2) * (mat ** 2)
        
    def calc_vel_bias (self, mat, index):

        return self.beta1 * self.bias_vel[index] + (1 - self.beta1) * mat

    def calc_rms_bias (self, mat, index):

        return self.beta2 * self.bias_rms[index] + (1 - self.beta2) * (mat ** 2)