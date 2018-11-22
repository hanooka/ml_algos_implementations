

class Layer:
    def __init__(self, dim):
        self.dim = dim

class FullyConnected(Layer):
    def __init__(self, dim, activation=None):
        super().__init__(dim)
        self.activation = activation

    def has_activation(self):
        return self.activation and str(type(self.activation)).startswith('activation')

    def set_activation(self, activation):
        self.activation = activation

class InputLayer(Layer):
    def __init__(self, dim):
        super().__init__(dim)
