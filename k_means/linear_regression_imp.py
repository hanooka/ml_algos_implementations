import numpy as np
import pandas as pd
from scipy.io import loadmat
#import mat4py
import h5py
import matplotlib.pyplot as plt

class MyLinearRegression(object):
    def __init__(self, alpha=0.1, lmbda=1.0):
        self.n = 0
        self.m = 0
        self.coeffs = None
        self.X = None
        self.y = None
        self.errors = None
        self.alpha = alpha
        self.lmbda = lmbda

    @staticmethod
    def plot_data(X, y):
        plt.figure()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression\nData Plot')
        plt.scatter(X, y, color = 'r', marker='x', s=20)

    def compute_cost(self):
        h_x = np.dot(self.X.T, self.coeffs)
        self.errors = h_x - self.y
        self.cost = np.sum(self.errors**2)/(2*self.m)
        reg_exp = self.lmbda*np.sum(self.coeffs**2)/(2*self.m)
        cost = np.sum(self.errors**2)/(2*self.m) + reg_exp
        print(cost)
        return cost

    def gradient_decent(self):
        before_cost = np.inf
        after_cost = self.compute_cost()
        while(abs(before_cost-after_cost) > 0.001):
            #print(after_cost)
            self.update_coeffs()
            before_cost = after_cost
            after_cost = self.compute_cost()
            if after_cost > before_cost:
                print("Failed to converge. alpha might be too big")
                quit()
        return self.cost

    def feature_normalize(self):
        pass

    def normal_eqn(self):
        pass

    def update_coeffs(self):
        for i, _ in enumerate(self.coeffs):
            if (i == 0):
                reg_exp = 0
            else:
                reg_exp = (self.lmbda*self.coeffs[i])/self.m
            self.coeffs[i] = self.coeffs[i] - (self.alpha * np.sum(self.errors * self.X.T[:, i])) / self.m

    def fit(self, X, y, Xval=None, yval=None):
        self.X = np.insert(X, 0, 1, 0)
        self.y = y
        self.m = X.shape[1]
        self.n = X.shape[0]
        #if self.coeffs is None:
        self.coeffs = np.random.uniform(-1, 1, self.n + 1)
            #self.coeffs = np.array([1, 1])

        after_cost = self.gradient_decent()
        print(self.coeffs)
        return after_cost, self.coeffs

    def pred(self, X):
        X = np.insert(X, 0, 1, 0)
        return np.dot(X.T, self.coeffs)


### Consts

f1_dfile = 'ex5data1.mat'

### Functions

def feature_normalize(X):
    for i in range(X.shape[0]):
        X[i] = (X[i] - X[i].mean()) / X[i].std()

def get_poly_features(base, degree=2):
    res = np.array(base)
    for i in range(degree-1):
        tmp = base
        tmp = get_multiplication_of_feature(tmp, base)
        res = np.append(res, tmp, axis=0)
    return (res.T)


def get_multiplication_of_feature(base, mult):
    return base*mult

def get_cost(y_true, y_pred):
    m = len(y_pred)
    return np.sum((y_pred-y_true)**2)/(2*m)

def main():
    lr = MyLinearRegression(alpha=0.001, lmbda=1)
    #df = pd.read_csv(f1_dfile, header=None)
    data = loadmat(f1_dfile)
    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']
    Xtest = data['Xtest']
    ytest = data['ytest']
    # LETS GO BBY
    data = np.asarray(data)

    X = X.T
    y = y.T

    X2 = get_poly_features(X, 8)
    X2 = X2.T
    feature_normalize(X2)

    Xval2 = get_poly_features(Xval.T, 8)
    Xval2 = Xval2.T
    feature_normalize(Xval2)
    #
    # lr.fit(X2, y)
    # y_pred = lr.pred(X2)
    # lr.plot_data(X, y)
    # plt.scatter(X.T, y_pred)
    # plt.show()

    errors = []
    eval_errors = []
    coeffss = []
    m_sizes = np.arange(2, 15, 1)
    for i in m_sizes:
        error, coeffs = lr.fit(X2[:,:i], y[:,:i])
        yval_pred = lr.pred(Xval2)
        eval_error = get_cost(yval.T, yval_pred)
        eval_errors.append(eval_error)
        errors.append(error)
        coeffss.append(coeffs)

    print(errors)
    print(eval_errors)

    plt.plot(m_sizes, errors, label='train_error')
    plt.plot(m_sizes, eval_errors, label='validation_error')
    plt.xlabel('training_size')
    plt.ylabel('error')
    plt.legend()
    plt.show()

    #MyLinearRegression.plot_data(X, y)

if __name__ == '__main__':
    main()