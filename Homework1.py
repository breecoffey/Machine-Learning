import numpy as np
import pandas as pd
from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split

# Reads the data from the Excel file into a DataFrame
trainingData = pd.read_csv('train.csv')

# Removes features that I didn't think were as useful
def RemoveUselessThings(df):
    return df.drop(['Ticket', 'Fare', 'Embarked', 'Name', 'Cabin'], axis = 1)

# Categorizes ages into bins
def GroupAges(df):
    df.Age = df.Age.fillna(-0.5)
    groupAges = (-1, 0, 6, 13, 20, 30, 60)
    groupTitles = ['N/A', 'Infant', 'Child', 'Teen' 'Young Adult', 'Adult', 'Senior']
    groups = pd.cut(df.Age, groupAges, labels=groupTitles)
    return df

#Applies the data transformation functions
def TransformData(df):
    df = RemoveUselessThings(df)
    df = GroupAges(df)
    return df

trainingData = TransformData(trainingData)

#Encodes the labels to the features
def EncodeLabels(dfTrain):
    features = ['Age', 'Sex', 'Pclass']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dfTrain[feature])
        dfTrain[feature] = le.transform(dfTrain[feature])

    return dfTrain

trainingData = EncodeLabels(trainingData)

#breaks up training data into x and y components
X_all = trainingData.drop(['Survived', 'PassengerId'], axis = 1)
y_all = trainingData['Survived']

#randomly splits the training data into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all)

#creates svm
classifier = svm.SVC(gamma=0.001)

#fits the training data to the model
classifier.fit(X_train, y_train)

#makes prediction on the test data
expectations = y_test
predictions = classifier.predict(X_test)

#prints out the accuracy score
print(metrics.accuracy_score(expectations, predictions))
