# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    N = y.size
    w = np.dot(np.linalg.inv(np.dot(tx.transpose(),tx)) ,np.dot(tx.transpose(),y))
    error = y-np.dot(tx,w)
    MSE = np.dot(error.transpose(),error )/(2*N)
    return MSE, w
