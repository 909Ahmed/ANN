import numpy as np
class Adam ():

    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999):
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = 1e-8
                
        self.weight_vel = [np.zeros((layer.size, layer.pre_layer.size)) for layer in layers[::-1] if layer.type != 'dropout']
        self.bias_vel = [np.zeros(layer.size) for layer in layers[::-1] if layer.type != 'dropout']

        self.weight_rms = [np.zeros((layer.size, layer.pre_layer.size)) for layer in layers[::-1] if layer.type != 'dropout']
        self.bias_rms = [np.zeros(layer.size) for layer in layers[::-1] if layer.type != 'dropout']


    def adam_weights (self, del_w, batch):
        
        self.weight_vel = [self.calc_vel_weights(mat, index) for index, mat in enumerate(del_w)]
        self.weight_rms = [self.calc_rms_weights(mat, index) for index, mat in enumerate(del_w)]

        vhat = [ele / (1 - (self.beta1 ** (batch + 1))) for ele in self.weight_vel]
        mhat = [ele / (1 - (self.beta2 ** (batch + 1))) for ele in self.weight_rms]

        return [self.lr * np.divide(vel, (np.sqrt(rms) + self.eps)) for vel, rms in zip(vhat, mhat)]


    def adam_bias (self, del_b, batch):

        self.bias_vel = [self.calc_vel_bias(mat, index) for index, mat in enumerate(del_b)]
        self.bias_rms = [np.add(self.bias_rms[index], self.calc_rms_bias(mat, index)) for index, mat in enumerate(del_b)]

        vhat = [ele / (1 - (self.beta1 ** (batch + 1))) for ele in self.bias_vel]
        mhat = [ele / (1 - (self.beta2 ** (batch + 1))) for ele in self.bias_rms]

        return [self.lr * np.divide(vel, (np.sqrt(rms) + self.eps)) for vel, rms in zip(vhat, mhat)]


    def calc_vel_weights (self, mat, index):

        return self.beta1 * self.weight_vel[index] + (1 - self.beta1) * mat

    def calc_rms_weights (self, mat, index):

        return self.beta2 * self.weight_rms[index] + (1 - self.beta2) * (mat ** 2)
        
    def calc_vel_bias (self, mat, index):

        return self.beta1 * self.bias_vel[index] + (1 - self.beta1) * mat

    def calc_rms_bias (self, mat, index):

        return self.beta2 * self.bias_rms[index] + (1 - self.beta2) * (mat ** 2)