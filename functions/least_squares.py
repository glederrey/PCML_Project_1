# -*- coding: utf-8 -*-
"""
Least Squares using normal equations
"""

import numpy as np
from functions.costs import *
    
def least_square(y, tx):
    """calculate the least squares solution."""    
    # Compute optimal weights
    xx = np.dot(np.transpose(tx),tx)        
    xy = np.dot(np.transpose(tx),y)
    w_star = np.linalg.solve(xx, xy)
    
    loss = compute_cost(y, tx, w_star, 'RMSE')
    
    return loss, w_star
