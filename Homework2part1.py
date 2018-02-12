import numpy as np
import matplotlib.pyplot as plt

X = np.array([[-2,4,-1],[4,1,-1],[1,6,-1],[2,4,-1],[6,2,-1],[0,3,-1],[0,1,-1],[2,7,-1],[6,4,-1],[-1,1,-1]])
y = np.array([-1,-1,1,1,1,-1,-1,1,1,-1])

X_nc = np.array([[2,2,-1],[4,4,-1],[3,3,-1],[1,1,-1],[1,2,-1],[3,4,-1],[5,5,-1],[1,5,-1],[2,3,-1],[4,1,-1]])
y_nc = np.array([-1,1,-1,1,1,1,-1,-1,1,-1,1])

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
ppn = Perceptron(eta = 0.1 ,n_iter=10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

ppn_nc = Perceptron(eta = 0.1, n_iter = 10)
ppn_nc.fit(X_nc, y_nc)
plt.plot(range(1, len(ppn_nc.errors_) + 1), ppn_nc.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

