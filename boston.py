#!/usr/bin/env python
# coding: utf-8
"""
author -- ToxaZ

Couresera Machine Learning Introduction 2nd week assignement
4 - https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/gXjdV/vybor-mietriki
"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from utils import write_submission, classifier_choice_cv


def main():
    boston = load_boston()
    X = scale(boston.data)
    y = boston.target
    classifier = KNeighborsRegressor(n_neighbors=5, weights='distance')
    kf = KFold(len(X), n_folds=5, shuffle=True, random_state=42)
    neighbors_range = np.linspace(1.0, 10.0, num=200)

    write_submission(
        int(classifier_choice_cv(
            X, y, classifier, 'p', neighbors_range, kf,
            scoring='mean_squared_error')[0]),
        '41')

if __name__ == '__main__':
    main()
