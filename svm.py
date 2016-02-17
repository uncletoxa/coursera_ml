#!/usr/bin/env python
# coding: utf-8
"""
author -- ToxaZ

Coursera Machine Learning Introduction 3nd week assignement
6 - https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/50VrR/opornyie-obiekty
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from utils import write_submission


def main():
    data = pd.read_csv('data/svm-data.csv', header=None)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    clf = SVC(C=100000, random_state=241)
    clf.fit(X, y)
    result = str([x+1 for x in clf.support_])
    write_submission(result, '61')

if __name__ == '__main__':
    main()
