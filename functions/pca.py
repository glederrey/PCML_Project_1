# -*- coding: utf-8 -*-
"""
Code for the PCA analysis
"""

import numpy as np

def mean_vector(tX):
    return np.mean(tX,axis=0)
    
def covariance_mat(tX):
    arrays = []
    for i in range(len(tX[0])):
        arrays.append(tX[i,:])
        
    return np.cov(arrays)