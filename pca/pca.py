import numpy as np




class PCA(object):
    """ Principal component analysis (PCA)
    {M, N} => {M, K} where K < N
    """


    def __init__(self, dimensions=2):
        self._dimensions = dimensions
        self._transformation_matrix = None

    def get_Sigma(self, X):
        return ((1/X.shape[0]) * np.dot(X, X.T))

    def get_normalized(self, X, normalization_method):
        if normalization_method == None:
            return X
        elif normalization_method == 'minmax':
            return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        elif normalization_method == 'std':
            return (X - X.mean(axis=0)) / X.std(axis=0)
        else:
            raise ValueError("no such normalization method: {}".format(normalization_method))

    def get_u_reduce(self, X):
        sigma = self.get_Sigma(X)
        u, s, v = np.linalg.svd(sigma)
        return u[:, 0:self._dimensions]

    def get_transformation_matrix(self, X, normalization_method):
        _X = self.get_normalized(X.copy(), normalization_method)
        return np.dot(self.get_u_reduce(_X).T, _X).T

    def fit(self, X, normalization_method=None):
        self._transformation_matrix = self.get_transformation_matrix(X, normalization_method)

    def fit_transform(self, X, normalization_method=None):
        self.fit(X, normalization_method)
        return(np.dot(X, self._transformation_matrix))

if __name__ == '__main__':
    # Comment
    pca = PCA(2)
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 5, 4], [1, 1, 2]])
    print("The Matrix: \n", mat, '\n')
    print("Transposed using Standardization: \n", pca.fit_transform(mat, 'std'), '\n')
    print("Transposed using MinMax Scale: \n", pca.fit_transform(mat, 'minmax'), '\n')
    print("Transposed without Scaling: \n", pca.fit_transform(mat), '\n')




