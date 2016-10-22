# -*- coding: utf-8 -*-
"""
Ridge Regression
"""

import numpy as np
from functions.costs import *
import matplotlib.pyplot as plt
from IPython import display
from matplotlib import cm
from functions.helpers import build_k_indices
    
def build_poly(x, degree):
    n_x = len(x)
    nbr_param = len(x[0])
    mat = np.zeros((n_x, (degree+1)*nbr_param))
        
    for j in range(nbr_param):
        for k in range(degree+1):
            mat[:, j*(degree+1)+k] = x[:,j]**k
            
    return mat     

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    
    # Compute optimal weights
    xx = np.dot(np.transpose(tx),tx)
   
    bxx = xx + lamb*np.identity(len(xx))
        
    xy = np.dot(np.transpose(tx),y)
    
    w_star = np.linalg.solve(bxx, xy)
    
    print(w_star)
    
    loss = compute_cost(y, tx, w_star, 'RMSE')
    
    return loss, w_star
    
def find_min(rmse_te, lambdas, degrees):
    print("Min for rmse_te: %f"%(np.min(rmse_te)))
    x, y = np.where(rmse_te == np.min(rmse_te))
    ilamb_star = x[0]
    ideg_star = y[0]
    print("test = %f"%(rmse_te[ilamb_star,ideg_star]))
    
    return lambdas[ilamb_star], int(degrees[ideg_star])
    
def cross_validation(y, tx, lambdas, degrees, k_fold, verb = False, seed = 1):
    """
        K-fold cross validation for the Ridge Regression
    """
    
    print("Start the %i-fold Cross Validation!"%k_fold)
    
    # Prepare the matrix of rmse
    rmse_te = np.zeros((len(lambdas), len(degrees)))  
    
    # Split data in k-fold  
    k_indices = build_k_indices(y, k_fold, seed)

    # Loop on the degrees    
    for ideg, deg in enumerate(degrees):
        deg = int(deg)   
        # Loop on the lambdas
        for ilamb, lamb in enumerate(lambdas):
            loss_te = []
            if verb:
                print("Degree: %i, Lambda: %f; loss = median("%(deg, lamb), end='')
            # Loop on the k indices
            for k in range(k_fold):
                loss = calculate_cv(y, tx, k_indices, k, lamb, deg)
                loss_te.append(loss)
                if verb:
                    if k < k_fold - 1:
                        print("%f, "%loss, end='')
                    else:
                        print("%f) = %f"%(loss, np.median(loss_te)))
            # We take the median in case of an outlier. 
            rmse_te[ilamb, ideg] = np.median(loss_te)
            
        print("Degree %i/%i done!"%(deg, degrees[-1])) 
        
    print("%i-fold Cross Validation finished!"%k_fold)           
            
    return rmse_te    

def calculate_cv(y, tx, k_indices, k, lamb, degree):
    # Values for the test 
    tx_test = tx[k_indices[k]]    
    y_test = y[k_indices[k]]   
    
    # Get all the indeices that are not in the test data
    #index_not_k = np.array([i for i in range(len(tx)) if i not in k_indices[k]])
    train_indices = []
    for i in range(len(k_indices)):
        if i != k:
            train_indices.append(k_indices[i])
            
    train_indices = np.array(train_indices)
    train_indices = train_indices.flatten()
    
    # Values for the train
    tx_train = tx[train_indices]
    y_train = y[train_indices]
    
    # Build the polynomials functions
    tX_train = build_poly(tx_train, degree)
    tX_test = build_poly(tx_test, degree)  
    
    # Apply the Ridge Regression
    _, w_star = ridge_regression(y_train, tX_train, lamb) 
    
    # Return the RMSE on the test data
    return compute_cost(y_test, tX_test, w_star, 'RMSE')      
