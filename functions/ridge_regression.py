# -*- coding: utf-8 -*-
"""
Ridge Regression
"""

import numpy as np
from functions.costs import *
import matplotlib.pyplot as plt
from IPython import display
from matplotlib import cm

def build_poly(x, degree):
    n_x = len(x)
    nbr_param = len(x[0])
    mat = np.zeros((n_x, (degree+1)*nbr_param))
        
    for i in range(n_x):
        for j in range(nbr_param):
            for k in range(degree+1):
                mat[i][j*(degree+1)+k] = x[i][j]**k
            
    return mat 
    
def build_poly_fast(x, degree): 
    result = np.zeros((x.shape[0],(degree+1)*x.shape[1])) 
    k = 0
    for d in range(0, degree+1):
        for j in range(x.shape[1]):
           # print (j+k, x[:,j]**d)
            result[:,j + k] = x[:,j]**d
        k += len(x[0])
    return result 

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""

    # Compute optimal weights
    xx = np.dot(np.transpose(tx),tx)
    
    bxx = xx + lamb*np.identity(len(xx))
        
    xy = np.dot(np.transpose(tx),y)
    w_star = np.linalg.solve(bxx, xy)
    
    loss = compute_cost(y, tx, w_star, 'RMSE')
    
    return loss, w_star
    
def cross_validation(y_train, x_train, y_test, x_test, lambdas, degrees):
    rmse_tr = np.zeros((len(lambdas), len(degrees)))
    rmse_te = np.zeros((len(lambdas), len(degrees)))    
    
    for ideg, deg in enumerate(degrees):
        deg = int(deg)
        tX_train = build_poly(x_train, deg)
        tX_test = build_poly(x_test, deg)    
        for ilamb, lamb in enumerate(lambdas):
            loss, w = ridge_regression(y_train, tX_train, lamb)
            #print("lambda = %f, degree = %f: rmse_tr = %f"%(lamb, deg, loss))
            rmse_tr[ilamb, ideg] = loss
            rmse_te[ilamb, ideg] = compute_cost(y_test, tX_test, w, 'RMSE')
            
        print("Degree %i/%i done!"%(deg, degrees[-1]))            
            
    return rmse_tr, rmse_te
    
def find_min(rmse_tr, rmse_te, lambdas, degrees):
    print("Min for rmse_te: %f"%(np.min(rmse_te)))
    x, y = np.where(rmse_te == np.min(rmse_te))
    ilamb_star = x[0]
    ideg_star = y[0]
    print("test = %f"%(rmse_te[ilamb_star,ideg_star]))
    
    return lambdas[ilamb_star], int(degrees[ideg_star])


