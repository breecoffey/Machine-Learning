class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

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
    cost_ : list
        Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

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
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression, we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        print(self.w_)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)    
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Reads the data from the Excel file into a DataFrame
trainingData = pd.read_csv('train.csv')

def RemoveUselessThings(df):
    return df.drop(['Ticket','Embarked','Name','Cabin'], axis = 1)

def FixDeath(df):
    df.Survived = df.Survived.replace(0,-1)
    return df
def FixAges(df):
    df.Age = df.Age.fillna(df['Age'].mean())
    df['Age'] = df['Age'].astype(int)
    return df

def FixFare(df):
    df['Fare'] = df['Fare'].astype(int)
    return df

def FixGender(df):
    df['Sex']=df['Sex'].replace('female', 1)
    df['Sex']=df['Sex'].replace('male', -1)
    return df

def CleanUpData(df):
    df = FixDeath(df)
    df = RemoveUselessThings(df)
    df = FixAges(df) 
    df = FixGender(df)
    df = FixFare(df)
    return df

trainingData = CleanUpData(trainingData)

#breaks up training data into x and y components
X_all = trainingData.iloc[0:890, [3]].values
y_all = trainingData.iloc[0:890, 1].values

#randomly splits the training data into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.30)

ada = AdalineGD(n_iter=200, eta=0.0001)
ada.fit(X_train, y_train)
expectations = y_test
predictions = ada.predict(X_test)
print(metrics.accuracy_score(expectations,predictions))


