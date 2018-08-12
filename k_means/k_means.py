import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

### Classes

class KMeans(object):
    def __init__(self, k=5):
        self.k = k
        self.centroids = None # mew
        self.distance_metric = None

    def __test_X_k(self, X):
        if X.shape[0] < self.k:
            raise ValueError("Number of examples is smaller then number of clusters")

    @staticmethod
    def euclidean_distance(vector_a, vector_b):
        """
        calculating euclidean distance between to vectors
        Parameters
        ----------
        vector_a: vector/array of shape (1, n)
        vector_b: vector/array of shape (1, n)

        returns: the euclidean distance between vector_a and vector_b
        """

        squared_sum = 0
        for a, b in zip(vector_a, vector_b):
            squared_sum += (a-b)**2
        return np.sqrt(squared_sum)

    def init_centroids(self, X):
        """
        Randomly init self.centroids by sampling X
        self.centroids will contain a list of the centroids

        param X: Training data

        returns: None
        """

        self.__test_X_k(X)
        idx = np.random.randint(X.shape[0], size=self.k)
        self.centroids = X[idx,:].copy()
        self.distance_metric = KMeans.euclidean_distance

    def getIndexOfClosestCentroid(self, x):
        """
        finding the closest centroid to the sample x

        param x: a sample of X (a vector)

        return: index of the centroid
        """

        min_distance = np.inf
        min_distance_index = -1
        for i, centroid in enumerate(self.centroids):
            distance = self.distance_metric(centroid, x)
            if min_distance > distance:
                min_distance = distance
                min_distance_index = i
        return min_distance_index

    def stepCenterOfCentroids(self, X):
        """
        This is the "step" method
        For each X[i] from X we will set C[i] as the index of the closest centroid to the sample
        Then, we will re-calculate the "center" of all centroids

        param X: Training data

        returns: None
        """

        C = []
        # Setting C with the indexes of the centroids.
        for x in X:
            C.append(self.getIndexOfClosestCentroid(x))
        C = np.array(C)
        for i in range(self.k):
            # Case a centroid isn't close to any sample, we will remove it and reduce k to k-1.
            if np.where(C == 4)[0].size == 0:
                self.centroids.pop(i)
                self.k =- 1
            else:
                centroid_indices = np.where(C == i)[0]
                self.centroids[i] = np.mean(X[centroid_indices], axis=0)


    def fit(self, X):
        """
        FIT ME
        :param X:
        :return:
        """
        pass


### Consts

data_dir = './data/'
f_ex1 = data_dir + 'ex7data1.mat'
f_ex2 = data_dir + 'ex7data2.mat'
f_faces = data_dir + 'ex7faces.mat'

### Functions


### Main

def main():
    data = loadmat(f_ex1)
    npdata = np.array(data['X'])
    print(npdata.shape)

if __name__ == '__main__':
    main()

