import numpy as np
import pandas as pd
import re
from sklearn import metrics
from collections import Counter

#This function creates a dataframe from the given text files. The df.sample() function shuffles the positive and negative reviews.
def create_dataframe(postive_file, negative_file):
    df_pos = pd.read_table(postive_file, '\n')
    df_neg = pd.read_table(negative_file, '\n')

    df_pos['sent'] = 1
    df_neg['sent'] = -1

    dfs = [df_pos, df_neg]
    df = pd.concat(dfs)
    df = df.sample(frac=1)
    return df

# This function removes the stopwords and punctuation using a lambda function
def remove_stop_words(data):
    stopwords = open("stopwords.txt").read().splitlines()
    data['reviews'] = data['reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    return data

# This function splits the review based on whitespace and counts up the occurrence of each word
def count_text(text):
    words = []
    for word in text:
        words += re.split("\s+", word)
    return Counter(words)

# This function generates the counts for the naive bayes classifier, returned in an array
def gen_counts(df):
    pos_all = df.loc[df['sent'] == 1]
    pos_X_all = np.array(pos_all.iloc[:, 0]) #All of the positive reviews

    neg_all = (df.loc[df['sent'] == -1])
    neg_X_all = np.array(neg_all.iloc[:, 0]) #All of the negative reivews

    pos_rev_count = len(pos_X_all)
    neg_rev_count = len(neg_X_all)

    prob_positive = pos_rev_count / len(df) #Probability of the positive class
    prob_negative = neg_rev_count / len(df) #Probablitiy of the negative class

    positive_counts = count_text(pos_X_all) #All of the counts of the words of pos class
    negative_counts = count_text(neg_X_all) #All of the counts of the words of neg class
    return[positive_counts, negative_counts, prob_positive, prob_negative, pos_rev_count, neg_rev_count]

# Generating probabilites for class prediction using the naive bayes formula
def make_class_prediction(text, counts, class_prob, class_count):
    prediction = 1 #Smoothing variable
    text_counts = Counter(re.split("\s+", text)) #generate counts of given text
    for word in text_counts:
        prediction *= text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
    return prediction * class_prob

# Calculates the probability for the positive and negative class, and returns the prediction of whichever is greater
def make_decision(text, counts):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, counts[1], counts[3], counts[5])
    positive_prediction = make_class_prediction(text, counts[0], counts[2], counts[4])

    # We assign a classification based on which probability is greater.
    if (negative_prediction > positive_prediction):
        return -1
    else:
        return 1

# Wrapper function that generates the expectations and predictions for calculating accuracy
def naive_bayes(train_df, test_df):
    train_df = remove_stop_words(train_df)
    test_df = remove_stop_words(test_df)
    train_counts = gen_counts(train_df)
    X_all = np.array(test_df.iloc[:, 0].values)
    y_expectations = np.array(test_df['sent'])
    y_predictions = []
    for x in X_all:
        y_predictions.append(make_decision(x, train_counts))
    return y_expectations, y_predictions


train_df = create_dataframe('rt-polarity-pos.txt', 'rt-polarity-neg.txt')
dev_df = create_dataframe('rt-polarity-pos-dev.txt', 'rt-polarity-dev-neg.txt')
values_for_accuracy = np.array(naive_bayes(train_df, dev_df))
accuracy_score = metrics.accuracy_score(values_for_accuracy[0], values_for_accuracy[1])
print("Development Accuracy: ", accuracy_score)

test_df = create_dataframe('rt-polarity-pos-test.txt','rt-polarity-test-neg.txt')
values_for_accuracy_test = np.array(naive_bayes(train_df, test_df))
accuracy_score_test = metrics.accuracy_score(values_for_accuracy_test[0], values_for_accuracy_test[1])
print("Test Accuracy: ", accuracy_score_test)