# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    #set seed
    np.random.seed(seed)
    bound = int(ratio*y.size)
    ind = np.random.permutation(len(y))
    return y[ind[:bound]], y[ind[bound:]], x[ind[:bound]], x[ind[bound:]]
