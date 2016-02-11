#!/usr/bin/env python
# coding: utf-8
"""
author -- ToxaZ

various helpers for submissions
"""
import numpy as np
from sklearn.cross_validation import cross_val_score


def write_submission(result, filename):
    '''write results in proper Coursera format'''
    with open('submissions/{}'.format(filename), 'w') as f:
            result = ''.join(c for c in str(result) if c not in '[](){}<>,\'')
            f.write(str(result))
    print '"{0}" has been written to submissions/{1}'.format(result, filename)


def classifier_choice_cv(X, y, classifier, parameter_name, parameter_range, cv,
                         scoring=None, print_scores=False):
    '''return best cv score for given parameter in given classifier'''
    cv_scores = []
    for parameter in parameter_range:
        scores = []
        classifier_var = classifier
        setattr(classifier_var, parameter_name, parameter)
        scores = cross_val_score(classifier_var, X, y, cv=cv, scoring=scoring)
        cv_scores.append(np.mean(scores))

    np_scores = np.fromiter(cv_scores, np.float)
    if print_scores:
        results = {}
        for x in range(len(parameter_range)):
            results[parameter_range[x]] = cv_scores[x]
        print results
    return [parameter_range[np_scores.argmax()], cv_scores[np_scores.argmax()]]
