from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np

'''
Implementation of KNN
'''

data = pd.read_csv('data.csv')

def rename_diagnosis(df):
    df['diagnosis'] = df['diagnosis'].replace('B', 1)
    df['diagnosis'] = df['diagnosis'].replace('M', 0)
    return df


def remove_id(df):
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    return df


data = rename_diagnosis(data)
data = remove_id(data)

X_all = np.array(data.iloc[0:570, 2:].values)
y_all = np.array(data['diagnosis'])

X_train, X_dev_test, y_train, y_dev_test = train_test_split(X_all, y_all, test_size=0.30)
y_temp = np.reshape(y_dev_test, len(y_dev_test))
X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_temp, test_size=0.50)

def knn_train(X_train, y_train):
    return


def knn_predict(X_train, y_train, x_dev, k):
    distances = []
    targets = []

    for i in range(len(X_train)):
        distance = np.sqrt(np.sum(np.square(x_dev - X_train[i, :])))
        distances.append([distance, i])

    distances = sorted(distances)

    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    return Counter(targets).most_common(1)[0][0]


def knn(X_train, y_train, X_dev, predictions, k):
    # train on the input data
    knn_train(X_train, y_train)

    # loop over all observations
    for i in range(len(X_dev)):
        predictions.append(knn_predict(X_train, y_train, X_dev[i, :], k))


predictions = []

knn(X_train, y_train, X_dev, predictions, 3)

predictions = np.asarray(predictions)

print(metrics.accuracy_score(y_dev, predictions))
