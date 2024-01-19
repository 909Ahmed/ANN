class Dropout():

    def __init__(self, rate, pre_layer):
        
        self.rate = rate
        self.pre_layer = pre_layer
        self.size = pre_layer.size
        self.type = 'dropout'