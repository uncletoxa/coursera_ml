#!/usr/bin/env python
# coding: utf-8
"""
author -- ToxaZ

Coursera Machine Learning Introduction 3nd week assignement
7 - https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/NdyLO/analiz-tiekstov
"""

from __future__ import unicode_literals
import numpy as np
import logging
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from utils import write_submission


def show_top10(classifier, vectorizer):
    feature_names = np.asarray(vectorizer.get_feature_names())
    top10 = np.argsort(
        np.absolute(
            np.asarray(
                classifier.coef_.todense()
            )
        ).reshape(-1))[-10:]
    return feature_names[top10].tolist()


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    newsgroups = fetch_20newsgroups(subset='all',
                                    categories=['alt.atheism', 'sci.space'])

    vectorizer = TfidfVectorizer()
    X = newsgroups.target
    y = newsgroups.data
    X_train = vectorizer.fit_transform(y)
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(X_train.shape[0], n_folds=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)

    gs.fit(X_train, X)
    clf.set_params(**gs.best_params_)
    clf.fit(X_train, X)
    result = (show_top10(clf, vectorizer))
    result.sort()
    write_submission(str(
            [x for x in result]).lower().encode('ascii', 'ignore'),
        '71')  # still need some work to get rid of unicode problem

if __name__ == '__main__':
    main()
