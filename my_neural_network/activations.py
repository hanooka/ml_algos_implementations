import numpy as np


class Activation:
    def __repr__(self):
        return ("{__class__.__name__}()".format(__class__=self.__class__))

    def get_activation_func(self):
        pass


class SigmoidActivation(Activation):
    def __init__(self):
        pass

    def __str__(self):
        return "sigmoid"

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def get_activation_func(self):
        return SigmoidActivation.sigmoid


class TanhActivation(Activation):
    def __init__(self):
        pass

    def __str__(self):
        return "tanh"

    @staticmethod
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def get_activation_func(self):
        return TanhActivation.tanh
