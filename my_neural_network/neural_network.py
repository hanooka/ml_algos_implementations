import numpy as np
from layers import *
from activations import *

class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.activations = []
        self.learning_rate = learning_rate

    def add_layer(self, layer):
        self.layers.append(layer)

    def initialize_network(self):
        pass

    def fit(self, X, y, verbose=False):
        pass

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass


if __name__ == '__main__':
    # Simulate a Xor problem
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([0, 1, 1, 0])

    # Building the network
    nn = NeuralNetwork()
    nn.add_layer(InputLayer(2))
    nn.add_layer(FullyConnected(3, SigmoidActivation()))
    nn.add_layer(FullyConnected(1, SigmoidActivation()))
    nn.initialize_network()
    nn.fit(X, y)

    preds = nn.predict(X)