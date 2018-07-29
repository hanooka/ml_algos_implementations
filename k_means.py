import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

### Classes

class KMeans(object):
    def __init__(self, k=5):
        self.k = k
        self.centroids = None

    def init_centroids(self, X):
        """
        Randomly init self.centroids by sampling X
        self.centroids will contain a list of the centroids
        :param X: Training data
        :return: None
        """
        pass

    def findClosestCentroid(self, x):
        """

        :param x:
        :return: index of the centroid
        """
        pass

    def findClosestCentroids(self, X):
        """
        for each x(i) from X we will set the result array res[i] = K - centroid
        :param X:
        :return: index array of centroids (len(x) == len(centroids))
        """
        pass


### Consts

f_ex1 = 'ex7data1.mat'
f_ex2 = 'ex7data2.mat'
f_faces = 'ex7faces.mat'

### Functions


### Main

def main():
    data = loadmat(f_ex1)
    npdata = np.array(data['X'])
    print(npdata.shape)

if __name__ == '__main__':
    main()

