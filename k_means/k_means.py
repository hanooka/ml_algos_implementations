import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

### Classes

class KMeans(object):
    def __init__(self, k=5):
        self.k = k
        self.centroids = None


    def __test_X_k(self, X):
        if X.shape[0] < self.k:
            raise ValueError("Number of examples is smaller then number of clusters")

    def init_centroids(self, X):
        """
        Randomly init self.centroids by sampling X
        self.centroids will contain a list of the centroids

        Parameters
        ----------

        param X: Training data

        Returns
        -------

        None
        """
        self.__test_X_k(X)
        idx = np.random.randint(X.shape[0], size=self.k)
        self.centroids = X[idx,:].copy()

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

