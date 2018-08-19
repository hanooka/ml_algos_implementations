import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

### Classes


class KMeans(object):
    def __init__(self, k=5, distance_metric=None, delete_centroid=False):
        self.k = k
        self.centroids = None # mew
        self.distance_metric = KMeans.getDistanceMetric(distance_metric)
        self.delete_centroid = delete_centroid

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

        return np.sqrt(np.sum((vector_a-vector_b)**2))

        # Can also
        # squared_sum = 0
        # for a, b in zip(vector_a, vector_b):
        #     squared_sum += (a-b)**2
        #
        # return np.sqrt(squared_sum)


    @classmethod
    def getDistanceMetric(cls, distance_metric):
        """
        Returning a distance metric method
        Supports: euclidean_distance
        """
        if distance_metric is None:
            return KMeans.euclidean_distance
        elif distance_metric == 'euclidean_distance':
            return KMeans.euclidean_distance
        else:
            raise ValueError("{} is not a valid distance_metric/unsupported.".format(distance_metric))

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

        returns: A list mC of the centroid VECTORS closest to the correlated sample
        """

        C = []  # Will contain the INDEX of the closest centroid for every given sample
        mC = [] # Will contain the VECTOR of the closest centroid for every given sample
        centroids_to_delete = [] # Will contain the indexes of centroids needed to be deleted

        # Setting C with the indexes of the centroids and mC with vectors of the centroids
        for x in X:
            C.append(self.getIndexOfClosestCentroid(x))
            mC.append(self.centroids[C[-1]])
        C = np.array(C)

        for i in range(self.k):
            # Case a centroid isn't close to any sample, we will remove it and reduce k to k-1.
            if self.delete_centroid and np.where(C == i)[0].size == 0 and self.k > 2:
                centroids_to_delete.append(i)
                self.k -= 1
            else:
                # Move centroid
                centroid_indices = np.where(C == i)[0]
                if centroid_indices.size > 0:
                    self.centroids[i] = np.mean(X[centroid_indices], axis=0)

        if self.delete_centroid:
            np.delete(self.centroids, centroids_to_delete)
        return mC

    def getCost(self, X, mC):
        return np.mean(np.sqrt(np.sum((X - mC)**2, axis=1)))

    def optimizeClusters(self, X):
        """ Optimize the values of the clusters """
        mC = self.stepCenterOfCentroids(X)
        cost = self.getCost(X, mC)
        for i in range(100000):
            new_mC = self.stepCenterOfCentroids(X)
            new_cost = self.getCost(X, new_mC)
            print("Step {}. Cost: {:.4f}".format(i, cost))
            if cost-new_cost < 0.0001:
                break
            else:
                cost = new_cost
        print("Finished with {} clusters\n".format(self.k))

    def fit(self, X):
        """ Fit(Optimize) the Algorithm Clusters with random init """

        self.init_centroids(X)
        self.optimizeClusters(X)

    def predict(self, X):
        """

        :param X:
        :return:
        """
        pass

### Consts

data_dir = './data/'
f_ex1 = data_dir + 'ex7data1.mat'
f_ex2 = data_dir + 'ex7data2.mat'
f_faces = data_dir + 'ex7faces.mat'
f_bird = data_dir + 'bird_small.mat'

### Functions


### Main


def reduce_image():
    data = loadmat(f_bird)
    npdata = np.array(data['A'])
    npdata = npdata.astype(np.int32)

    npdata_reshaped = npdata.reshape(128*128, 3)
    print(npdata_reshaped.shape)

    kmeans2 = KMeans(k=16)
    kmeans2.fit(npdata_reshaped)

def main():
    reduce_image()
    quit()

    data = loadmat(f_ex2)
    npdata = np.array(data['X'])
    # Printing first 6 samples
    print(npdata[:6])
    # Shape of data (samples, features)
    print(npdata.shape)

    # Running k means with k = 2, 3, 4, 5, 6, 7
    # We can see that moving from 2 clusters to 3 clusters reduce the "cost" significantly
    # But changing it from 3 to 4/5/6/7 reduce the "cost" insignificantly
    # Thus we can assume the problem can be solved with 3 clusters.

    kmeans2 = KMeans(k=1)
    kmeans2.fit(npdata)

    kmeans2 = KMeans(k=2)
    kmeans2.fit(npdata)

    kmeans3 = KMeans(k=3)
    kmeans3.fit(npdata)

    kmeans4 = KMeans(k=4)
    kmeans4.fit(npdata)

    kmeans5 = KMeans(k=5)
    kmeans5.fit(npdata)

    kmeans6 = KMeans(k=6)
    kmeans6.fit(npdata)

    kmeans7 = KMeans(k=7)
    kmeans7.fit(npdata)

if __name__ == '__main__':
    main()