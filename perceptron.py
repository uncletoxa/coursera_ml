#!/usr/bin/env python
# coding: utf-8
"""
author -- ToxaZ

Coursera Machine Learning Introduction 2nd week assignement
5 - https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/w7Rqc/normalizatsiia-priznakov
"""
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from utils import write_submission


def get_accuracy(X_train, y_train, X_test, y_test):
    clf = Perceptron(random_state=241)
    clf.fit(X_train, y_train)
    y_predictions = clf.predict(X_test)

    return accuracy_score(y_test, y_predictions)


def main():
    data = {}
    scaler = StandardScaler()

    for data_type in ['train', 'test']:
        df = pd.read_csv('perceptron-{0}.csv'.format(data_type), header=None)
        data['X_' + data_type] = df.iloc[:, 1:].values
        data['y_' + data_type] = df.iloc[:, 0].values

    data['X_train_scaled'] = scaler.fit_transform(data['X_train'])
    data['X_test_scaled'] = scaler.transform(data['X_test'])

    acc = get_accuracy(data['X_train'], data['y_train'],
                       data['X_test'], data['y_test'],)
    acc_scaled = get_accuracy(data['X_train_scaled'], data['y_train'],
                              data['X_test_scaled'], data['y_test'])

    write_submission(round(abs(acc - acc_scaled), 3), '51')

if __name__ == '__main__':
    main()
