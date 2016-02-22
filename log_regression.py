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


# splitting weights updater to 2 different functions is easier to handle
def update_weights(X, y, weights, k):
    dp = np.einsum('ij,j->i', X, weights[-1])
    c_matrix = 1-(1./(1+np.exp(-y*dp)))
    a = k/c_matrix.shape[0]*(np.einsum('ij,i,i->j', X, y, c_matrix))
    weights_list = []
    for w in xrange(len(weights[0])):
        weights_list.append(weights[-1][w] + a[w])
    weights.append(weights_list)
    return weights


def update_weights_reg(X, y, weights, c, k):
    dp = np.einsum('ij,j->i', X, weights[-1])
    c_matrix = 1-(1./(1+np.exp(-y*dp)))
    a = k/c_matrix.shape[0]*(np.einsum('ij,i,i->j', X, y, c_matrix))
    weights_list = []
    for w in xrange(len(weights[0])):
        weights_list.append(weights[-1][w] + a[w] - k*c*weights[-1][w])
    weights.append(weights_list)
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
    e_convergence = 0.00001
    max_iteration = 10000
    c_reg = 10
    k_step = 0.1
    weights_start = [[0.0, 0.0]]

    euclidean_distance = maxint
    iteration_count = 0
    while (math.sqrt(euclidean_distance) > e_convergence and
            iteration_count < max_iteration):
        weights_gradient = update_weights(X, y, weights_start, k_step)

        euclidean_distance = 0
        for w in xrange(len(weights_gradient[0])):
            euclidean_distance += (weights_gradient[-1][w] -
                                   weights_gradient[-2][w]) ** 2
        iteration_count += 1
    final_w = weights_gradient[-1]

    euclidean_distance = maxint
    iteration_count = 0
    while (math.sqrt(euclidean_distance) > e_convergence and
            iteration_count < max_iteration):
        weights_gradient_reg = update_weights_reg(X, y, weights_start,
                                                  c_reg, k_step)

        euclidean_distance = 0
        for w in xrange(len(weights_gradient_reg[0])):
            euclidean_distance += (weights_gradient_reg[-1][w] -
                                   weights_gradient_reg[-2][w]) ** 2
        iteration_count += 1
    final_w_reg = weights_gradient_reg[-1]

    y_scores = [sigmoid(X[i].tolist(), final_w) for i in xrange(len(X))]
    y_scores_reg = [sigmoid(X[i].tolist(), final_w_reg) for i in xrange(len(X))]

    write_submission(
        [round(roc_auc_score(y, y_scores), 3),
         round(roc_auc_score(y, y_scores_reg), 3)],
        '81')

if __name__ == '__main__':
    main()
