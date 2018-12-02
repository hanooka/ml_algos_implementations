import numpy as np
from layers import *
from activations import *


class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.weights = []
        self.biases = []
        self.learning_rate = learning_rate

    __hash__ = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def initialize_network(self, init_fact=0.1):
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.randn(len(self.layers[i]), len(self.layers[i+1]))*init_fact)
            self.biases.append(np.zeros(len(self.layers[i+1])))

    def feed_forward(self, X, y):
        A = [X]
        Z = []
        for i in range(len(self.layers)-1):
            A.append( np.dot(A[i], self.weights[i]) + self.biases[i])
            Z.append(self.layers[i+1].get_activation_func()(A[i+1]))

        return A, Z



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

    for i, j in zip(nn.weights, nn.biases):
        print(i, j)

    a, z = nn.feed_forward(X, y)
    for res in a:
        print(res, '\n')

    for res in z:
        print(res, '\n')

    quit()

    nn.fit(X, y, verbose=True)

    preds = nn.predict(X)
