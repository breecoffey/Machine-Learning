import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split

#read in the data file into a pandas dataframe
data = pd.read_csv('data.csv')

#replaces the benign classification with 1, and the malignant classification with zero
def rename_diagnosis(df):
    df['diagnosis'] = df['diagnosis'].replace('B', 1)
    df['diagnosis'] = df['diagnosis'].replace('M', 0)
    return df

#removes the two columns ID and Unnamed: 32 which are irrelevant to the model
def remove_id(df):
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    return df

#performs the transformations
data = rename_diagnosis(data)
data = remove_id(data)

#separates the dataframe into X and y values for each example where y is the class label and X is the feature label for each example
X_all = data.drop(['diagnosis'], axis=1)
y_all = data['diagnosis']

#normalizes the feature data
X_all = preprocessing.normalize(X_all, norm='l2')

#this plot displays the number of benign(1) and the number of malignant(0) classifications
sns.countplot(data['diagnosis'],label="Count")
plt.show()

#create a list of all features in order to create plots
features_mean=list(data.columns[1:11])

# split data frame into two based on diagnosis
dfM=data[data['diagnosis'] ==1]
dfB=data[data['diagnosis'] ==0]

#Stack the data into histograms to show most predictive features of classification
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth = (max(data[features_mean[idx]]) - min(data[features_mean[idx]]))/50
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(data[features_mean[idx]]), max(data[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, normed = True, label=['M','B'],color=['r','b'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()

# the histograms reveal that there are no outliers which are significantly affecting the outcome of prediction,
# so no further cleanup is really needed

#split data into np arrays
X_all = np.array(data.iloc[0:570, 2:].values)
y_all = np.array(data['diagnosis'])

#split data into training, developing, and testing (reshape development and test y values array)
X_train, X_dev_test, y_train, y_dev_test = train_test_split(X_all, y_all, test_size=0.30)
y_temp = np.reshape(y_dev_test, len(y_dev_test))
X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_temp, test_size=0.50)

#create the SVM classfier
classifier = svm.SVC(gamma=.001, random_state=1)

#fit classifier to data
classifier.fit(X_train, y_train)

#predict on the development data
expectations = y_dev
predictions = classifier.predict(X_dev)

#print the accuracy score
print("SVM on development accuracy: ")
print(metrics.accuracy_score(expectations, predictions))

# no parameter tweaks, around 65% accuracy
# adding random state = 1, around 70% accuracy
# setting gamma to .001, around 92% accuracy

#training the knn model function, returns nothing because there is no training in knn
def knn_train(X_train, y_train):
    return

#knn model prediction function
def knn_predict(X_train, y_train, x_dev, k):
    #two empty arrays, one for distances and the other for labels
    distances = []
    labels = []

    #iterates through the features and calculates the Euclidean distance
    for i in range(len(X_train)):
        distance = np.sqrt(np.sum(np.square(x_dev - X_train[i, :])))
        distances.append([distance, i])

    #sort the distances in ascending order
    distances = sorted(distances)

    #create a list of the k nearest neighbor's y labels
    for i in range(k):
        index = distances[i][1]
        labels.append(y_train[index])

    #returns the most common y label
    return Counter(labels).most_common(1)[0][0]


def knn(X_train, y_train, X_dev, predictions, k):
    # train on the input data
    knn_train(X_train, y_train)

    # loop over all observations
    for i in range(len(X_dev)):
        predictions.append(knn_predict(X_train, y_train, X_dev[i, :], k))


predictions = []

knn(X_train, y_train, X_dev, predictions, 3)

predictions = np.asarray(predictions)

print("KNN on development accuracy: ")
print(metrics.accuracy_score(y_dev, predictions))

#SVM ON TEST DATA
expectations = y_test
predictions = classifier.predict(X_test)

#print the accuracy score
print("SVM on test accuracy: ")
print(metrics.accuracy_score(expectations, predictions))

#KNN ON TEST DaTA
predictions = []

knn(X_train, y_train, X_test, predictions, 3)

predictions = np.asarray(predictions)

print("KNN on test accuracy: ")
print(metrics.accuracy_score(y_test, predictions))