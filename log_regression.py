#!/usr/bin/env python
# coding: utf-8
"""
author -- ToxaZ
credit more intelligent weights updating solution to Stanislav Zaluzhsky

Coursera Machine Learning Introduction 3nd week assignement
8 - https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/MptFX/loghistichieskaia-rieghriessiia
"""
import pandas as pd
import numpy as np
import math
from sys import maxint
from sklearn.metrics import roc_auc_score
from utils import write_submission


def create_gradient_descent(X, y, weights,
                            e_distance, iterations_count=maxint,
                            reg_coef=0, k_step=0, regularization=False):

    def update_weights(X, y, weights, reg_coef, k_step, regularization):
        dp = np.einsum('ij,j->i', X, weights[-1])
        c_matrix = 1 - (1. / (1 + np.exp(-y * dp)))
        a = k_step/c_matrix.shape[0]*(np.einsum('ij,i,i->j', X, y, c_matrix))
        if regularization is False:
            reg_coef = 0
        weights_list = []
        for w in xrange(len(weights[0])):
            weights_list.append(weights[-1][w] + a[w] -
                                k_step * reg_coef * weights[-1][w])
        weights.append(weights_list)
        return weights

    e = 10000
    for step in xrange(iterations_count):
        while math.sqrt(e) > e_distance:
            weights = update_weights(X, y, weights,
                                     reg_coef, k_step, regularization)
            for w in xrange(len(weights[0])):
                e += (weights[-1][0] - weights[-2][0]) ** 2
    return weights


def sigmoid(X, coef):
    if len(X) != len(coef):
        raise ValueError('lengths of X and its coefficients are different')
    c = 0
    for i in range(len(X)):
        c += -X[i] * coef[i]
    return 1 / (1 + math.exp(c))


def main():
    data = np.genfromtxt('data/data-logistic.csv', delimiter=",")
    y = data[:, 0]
    X = data[:, 1:]
    reg_coef = 10
    reg_step = 0.1
    start_weights = [[0.0, 0.0]]
    euclidean_distance = 0.00001
    iterations_max = 10000

    weights_matrix = create_gradient_descent(
        X, y, start_weights,
        euclidean_distance, iterations_max
    )
    reg_weights_matrix = create_gradient_descent(
        X, y, start_weights,
        euclidean_distance, iterations_max,
        reg_coef, reg_step, regularization=True
    )

    y_scores = [sigmoid(X[i], weights_matrix[-1]) for i in xrange(len(X))]
    y_scores_reg = [sigmoid(X[i], reg_weights_matrix[-1]) for i in xrange(len(X))]
    write_submission(
        [round(roc_auc_score(y, y_scores), 3),
         round(roc_auc_score(y, y_scores_reg), 3)],
        '81')

if __name__ == '__main__':
    main()
