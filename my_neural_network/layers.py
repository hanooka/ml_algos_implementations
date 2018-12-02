from activations import *
from collections import defaultdict


class Layer:
    def __init__(self, dim):
        self.dim = dim

    def __len__(self):
        return self.dim

    def __repr__(self):
        return "{__class__.__name__}(dim={dim!r})".format(__class__=self.__class__, **self.__dict__)


class FullyConnected(Layer):
    def __init__(self, dim, activation='sigmoid'):
        super().__init__(dim)
        self.set_activation(activation)

    def __repr__(self):
        return "{__class__.__name__}(dim={dim!r}, activation={activation!s})".format(
            __class__=self.__class__, **self.__dict__)

    def has_activation(self):
        return self.activation and str(type(self.activation)).startswith("<class 'activation")


    def set_activation(self, activation):
        activation_map = {
            'sigmoid': SigmoidActivation(),
            'tanh': TanhActivation()
        }
        if isinstance(activation, str):
            self.activation = activation_map[activation]
        elif isinstance(activation, Activation):
            self.activation = activation
        else:
            raise AttributeError("{} is of type {}. expected {} or {}".format(activation, type(activation), str, Activation))

    def get_activation_func(self):
        return self.activation.get_activation_func()


class InputLayer(Layer):
    def __init__(self, dim):
        super().__init__(dim)


if __name__ == '__main__':
    # Writing local tests before creating a testing unit
    fc = FullyConnected(5, 'sigmoid')
