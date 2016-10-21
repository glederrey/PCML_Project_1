# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lamb):
    N = y.size
    w = np.dot(np.linalg.inv(np.dot(tx.transpose(),tx)+lamb*np.identity(tx.shape[1])*(2*N)) ,np.dot(tx.transpose(),y))
    error = y - np.dot(tx,w)
    MSE = np.dot(error.transpose(),error )/(2*N)
    return MSE, w

