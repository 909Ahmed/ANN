import numpy as np
from .neurons import Neurons

class Conv():


    def __init__(self, feature_maps, size):

        self.kernel = [Neurons(size ** 2) for _ in range(feature_maps)]
        self._maps = feature_maps