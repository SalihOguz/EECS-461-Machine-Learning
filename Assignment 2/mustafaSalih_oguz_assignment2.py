# -*- coding: utf-8 -*-
import numpy as np
import pickle


# PART a)Feature Extraction
def feature1(x):
    """This feature computes the proportion of black squares to the
       total number of squares in the grid.
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature1_value: type-float
       """
    feature1_value = 0.0
    black_count = 0  # sum of all black squares
    square_count = 0  # number of all items

    for i in x:
        for j in i:
            square_count += 1
            black_count += j

    feature1_value = black_count / square_count
    return feature1_value


def feature2(x):
    """This feature computes the sum of the max of continuous black squares
       in each row
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature2_value: type-float
       """
    feature2_value = 0.0  # sum of continuous black tiles

    for i in x:
        row_sum = 0
        temp_sum = 0
        for j in i:
            if j == 1:  # if tile is black
                temp_sum += 1
            else:  # if tile is not black
                if temp_sum > row_sum:
                    row_sum = temp_sum
                temp_sum = 0

        if temp_sum > row_sum:
            row_sum = temp_sum

        feature2_value += row_sum

    return feature2_value


# PART b) Preparing Data
def part_b():
    train_positives = pickle.load(open('training_set_positives.p', 'rb'))
    train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))

    lis = []
    for i in train_positives:
        lis.append((feature1(train_positives[i]), feature2(train_positives[i])))
    for i in train_negatives:
        lis.append((feature1(train_negatives[i]), feature2(train_negatives[i])))

    X = np.array(lis)
    lis = [1] * 150 + [0] * 150
    y = np.array(lis)
    return X, y


# PART c) Classification with SGDClassifier
def part_c(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """
    predicted_class = None
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(random_state=0, alpha=0.001, n_iter=20)
    X1, y = part_b()
    sgd_clf.fit(X1, y)
    values = np.array([(feature1(x), feature2(x))])
    ab = sgd_clf.predict(values)
    predicted_class = ab[0]

    return predicted_class


# PART d) Assess the performance of the classifier in part c
def part_d():
    from sklearn.metrics import confusion_matrix, precision_score, recall_score

    train_positives = pickle.load(open('training_set_positives.p', 'rb'))
    train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))
    all_matrix = []
    for i in train_positives:
        all_matrix.append(train_positives[i])
    for i in train_negatives:
        all_matrix.append(train_negatives[i])

    X, y_true = part_b()
    y_pred = []
    for i in all_matrix:
        y_pred.append(part_c(i))

    confusion_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return [precision, recall, confusion_matrix]


# PART e) Classification with RandomForestClassifier
def part_e(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """
    from sklearn.ensemble import RandomForestClassifier
    predicted_class = None
    rf_clf = RandomForestClassifier(random_state=0)
    X1, y = part_b()
    rf_clf.fit(X1, y)
    values = np.array([(feature1(x), feature2(x))])
    ab = rf_clf.predict(values)
    predicted_class = ab[0]

    return predicted_class


# PART f) Assess the performance of the classifier in part e
def part_f():
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    train_positives = pickle.load(open('training_set_positives.p', 'rb'))
    train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))
    all_matrix = []
    for i in train_positives:
        all_matrix.append(train_positives[i])
    for i in train_negatives:
        all_matrix.append(train_negatives[i])

    X, y_true = part_b()
    y_pred = []
    for i in all_matrix:
        y_pred.append(part_e(i))

    confusion_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return [precision, recall, confusion_matrix]


# PART g) Your Own Classification Model
def custom_model(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """
    from sklearn.ensemble import RandomForestClassifier
    predicted_class = None
    rf_clf = RandomForestClassifier(random_state=59)
    X1, y = data_prep()
    rf_clf.fit(X1, y)
    values = np.array([(feature1(x), feature2(x), feature3(x))])
    ab = rf_clf.predict(values)
    predicted_class = ab[0]

    return predicted_class

def custom_model_test():
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    train_positives = pickle.load(open('training_set_positives.p', 'rb'))
    train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))
    all_matrix = []
    for i in train_positives:
        all_matrix.append(train_positives[i])
    for i in train_negatives:
        all_matrix.append(train_negatives[i])

    X, y_true = data_prep()
    y_pred = []
    for i in all_matrix:
        y_pred.append(custom_model(i))

    confusion_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return [precision, recall, confusion_matrix]

def feature3(x):
    feature3_value = 0.0  # sum of continuous black tiles

    for i in range(len(x[0])):
        column_sum = 0
        temp_sum = 0
        for j in range(len(x)):
            if x[i][j] == 1:  # if tile is black
                temp_sum += 1
            else:  # if tile is not black
                if temp_sum > column_sum:
                    column_sum = temp_sum
                temp_sum = 0
        if temp_sum > column_sum:
            column_sum = temp_sum

        feature3_value += column_sum

    return feature3_value

def data_prep():
    train_positives = pickle.load(open('training_set_positives.p', 'rb'))
    train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))

    lis = []
    for i in train_positives:
        lis.append((feature1(train_positives[i]), feature2(train_positives[i]), feature3(train_positives[i])))
    for i in train_negatives:
        lis.append((feature1(train_negatives[i]), feature2(train_negatives[i]), feature3(train_negatives[i])))

    X = np.array(lis)
    lis = [1] * 150 + [0] * 150
    y = np.array(lis)
    return X, y
