#!/usr/bin/env python
# coding: utf-8
"""
author -- ToxaZ

Coursera Machine Learning Introduction 2nd week assignement
3 - https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/eblyU/vybor-chisla-sosiediei
"""
import pandas as pd
import io
import requests
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale
from utils import write_submission, classifier_choice_cv


def main():
    wine_file = 'wine.data'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    try:
        data = pd.read_csv(wine_file, header=None)
    except IOError:
        print ('No {0} file found, downloading it from {1}'
               .format(wine_file, url))
        wine_reques = requests.get(url).content
        data = pd.read_csv(io.StringIO(wine_reques.decode('utf-8')),
                           header=None)

    X = data.iloc[:, 1:].values
    X_scaled = scale(X)
    y = data.iloc[:, 0].values
    kf = KFold(len(data.index), n_folds=5, shuffle=True, random_state=42)
    neighbors_range = [x for x in range(1, 51)]

    write_submission(
        classifier_choice_cv(
            X, y, KNeighborsClassifier, 'n_neighbors', neighbors_range, kf)[0],
        '31')
    write_submission(
        classifier_choice_cv(
            X, y, KNeighborsClassifier, 'n_neighbors', neighbors_range, kf)[1],
        '32')
    write_submission(
        classifier_choice_cv(
            X_scaled, y, KNeighborsClassifier, 'n_neighbors', neighbors_range, kf)[0],
        '33')
    write_submission(
        classifier_choice_cv(
            X_scaled, y, KNeighborsClassifier, 'n_neighbors', neighbors_range, kf)[1],
        '34')

if __name__ == '__main__':
    main()
