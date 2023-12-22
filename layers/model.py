# from Input import Input
# from layer import Layer

class Model():

    def __init__(self, input_layer, output_layer):
        
        self.input_layer = input_layer,
        self.output_layer = output_layer,
        self.layers = self.get_layers()


    def forward_pass (self, sample):

        pre_layer = self.input_layer
        pre_layer.values = sample

        for layer in self.layers:
            layer.update(pre_layer.values)
            pre_layer = layer
        

    def get_layers (self):
        
        layers = [self.output_layer]
        curr_layer = self.output_layer
        
        while curr_layer != self.input_layer:

            layers.append(curr_layer)
            curr_layer = curr_layer.pre_layer

        # layers.append(curr_layer)
        return layers[::-1]