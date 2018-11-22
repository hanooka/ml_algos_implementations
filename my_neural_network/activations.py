import numpy as np

class Activation:
    pass

class SigmoidActivation(Activation):
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class TanhActivation(Activation):
    def __init__(self):
        pass

    @staticmethod
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))