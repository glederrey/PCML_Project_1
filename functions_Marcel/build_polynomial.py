# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np



def build_poly(x, degree):
    tx = np.ones([x.shape[0],degree+1])
    for ii in range(1,degree+1):  
            tx[:,ii] = np.power(x,ii).ravel()
    return tx
