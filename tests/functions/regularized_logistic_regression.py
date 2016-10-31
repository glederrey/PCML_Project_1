# -*- coding: utf-8 -*-
"""
Regularized Logistic Regression using Gradient Descent
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from functions.helpers import *
from functions.costs import *

def calculate_loss_reg_logit(y, tx, w, lambda_):
    return calculate_loss_logit(y, tx, w) + lambda_*np.linalg.norm(w)**2

def penalized_logistic_regression(y, tx, w, lambda_):
    """
        Return the Loss and the gradient of the regularized logistic regression
    """
    loss = calculate_loss_reg_logit(y, tx, w, lambda_)
    grad = calculate_gradient_logit(y, tx, w) + 2*lambda_*w
    return loss, grad

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
        Do one step of gradient descent, using the penalized logistic regression.
        Return the loss and updated w.
        """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w_new = w - gamma*gradient
    return loss, w_new
	
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, verbose=False):
    """
        Use the Logistic Regression method to find the best weights
        
        INPUT:
            y           - Predictions
            tx          - Samples
            initial_w   - Initial weights
            max_iters   - Maximum number of iterations
            gamma       - Step size
            
        OUTPUT:
            w           - Best weights
            loss        - Minimum loss
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [calculate_loss_reg_logit(y, tx, initial_w, lambda_)]
    w = initial_w
    iterations = [] 
       
    last_loss = 0
    
    for n_iter in range(max_iters):
        gma = gamma
        # Gradient descent method
        loss, w_test = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        while calculate_loss_reg_logit(y, tx, w_test, lambda_) > losses[-1] and gma > 1e-10:
            gma = gma/5.
            loss, w_test = learning_by_penalized_gradient(y, tx, w, gma, lambda_)
        
        w = w_test
            
        if loss==np.inf:
            break

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        if verbose:
            if n_iter % 100 == 0:
                print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))
                last_loss = loss
            
        # Stopping criteria for the convergence
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 1e-8:
            break
    if verbose:
        print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))     
    
    return ws[-1], losses[-1]	
   
def cross_validation(y, tx, deg_lambdas, degrees, gamma, max_iter, k_fold, digits, verbose = True, seed = 1):
    """
        K-fold cross validation for the Logistic Regression
    """
    
    assert digits>0, 'digits must be at least 1'
    if verbose:
        print("  Start the %i-fold Cross Validation!"%k_fold)
    
    # Prepare the matrix of rmse
    rmse_te = np.zeros(len(degrees))
    lambs_star = np.zeros(len(degrees)) 
    
    # Split data in k-fold  
    k_indices = build_k_indices(y, k_fold, seed)

    # Loop on the degrees    
    for ideg, deg in enumerate(degrees):
        if verbose:
            print("  Start degree %i"%(deg))
        deg = int(deg)
        
        # Create the matrices
        mats_train, pred_train, mats_test, pred_test = create_matrices(y, tx, k_indices, deg)
        
        # First, we find the best lambdas in the first digit
        size = len(deg_lambdas)
        rmse_lmbd = np.zeros(size)
        lmbd = np.zeros(size)
        
        idx = 0
        # Loop on the degrees of lambdas
        for idlamb, dlamb in enumerate(deg_lambdas):
            if verbose:
                print("    Power of lambda: %i"%dlamb)
            # loop on the first digit
            lambda_ = 10**int(dlamb)
            lmbd[idx] = lambda_
            
            loss_te = []
            # Loop on the k indices
            for k in range(k_fold):
                w_star, loss = reg_logistic_regression(pred_train[k], mats_train[k], lambda_, np.zeros(mats_train[k].shape[1]), max_iter, gamma) 
                loss_te.append(perc_wrong_pred_logit(pred_test[k], mats_test[k], w_star)) 
            
            rmse_lmbd[idx] = np.mean(loss_te)
            idx += 1
                        
        for dg in range(1, digits+1):
            if verbose:
                print("  Start for digit %i"%dg)
            
            idx_min = np.argmin(rmse_lmbd)
            
            if dg > 1:    
                if idx_min == 0:
                   rmse_lmbd = np.zeros(11)
                   lmbd = np.linspace(lmbd[0], lmbd[1], 11) 
                    
                elif idx_min == len(rmse_lmbd)-1:
                   rmse_lmbd = np.zeros(11)
                   lmbd = np.linspace(lmbd[-2], lmbd[-1], 11)
                else:
                    rmse_lmbd = np.zeros(21)
                    lmbd = np.linspace(lmbd[idx_min-1], lmbd[idx_min+1], 21)
            else:
                if idx_min == 0:
                   rmse_lmbd = np.zeros(5)
                   lmbd = np.linspace(lmbd[0], 0.5*lmbd[1], 5) 
                    
                elif idx_min == len(rmse_lmbd)-1:
                   rmse_lmbd = np.zeros(5)
                   lmbd = np.linspace(5*lmbd[-2], lmbd[-1], 5)
                else:
                    rmse_lmbd = np.zeros(10)
                    lmbd = np.append(np.linspace(5*lmbd[idx_min-1], lmbd[idx_min], 5, endpoint = False), np.linspace(lmbd[idx_min], 0.5*lmbd[idx_min+1], 5))
                             
            for ilmbd in range(len(lmbd)):
                print("    Testing lambda = %10.0e"%lmbd[ilmbd])
                loss_te = []
                # Loop on the k indices
                for k in range(k_fold):
                    w_star, loss = reg_logistic_regression(pred_train[k], mats_train[k], lmbd[ilmbd], np.zeros(mats_train[k].shape[1]), max_iter, gamma)
                    loss_te.append(perc_wrong_pred_logit(pred_test[k], mats_test[k], w_star))                         
                
                rmse_lmbd[ilmbd] = np.mean(loss_te)
                
        idx_min = np.argmin(rmse_lmbd)
        if verbose:
            print("  Finished Degree %i. Best lambda is %10.3e with percentage wrong pred %f"%(deg, lmbd[idx_min], rmse_lmbd[idx_min]))
        rmse_te[ideg] = rmse_lmbd[idx_min]
        lambs_star[ideg] = lmbd[idx_min]
        
        if verbose:
            print("  --------------------")
        
    if verbose:  
        print("%  i-fold Cross Validation finished!\n"%k_fold)    

    idx_min = np.argmin(rmse_te)
    lambda_star = lambs_star[idx_min]
    degree_star = degrees[idx_min]
    min_loss = rmse_te[idx_min]    
            
    return min_loss, degree_star, lambda_star   

def create_matrices(y, tx, k_indices, degree):
    mats_test = []
    pred_test = []
    mats_train = []
    pred_train = []
    for k in range(len(k_indices)):
        # Values for the test 
        tx_test = tx[k_indices[k]]    
        pred_test.append(y[k_indices[k]]) 
        
        # Get all the indices that are not in the test data
        train_indices = []
        for i in range(len(k_indices)):
            if i != k:
                train_indices.append(k_indices[i])
                
        train_indices = np.array(train_indices)
        train_indices = train_indices.flatten()
        
        # Values for the train
        tx_train = tx[train_indices]
        pred_train.append(y[train_indices])
        
        if degree == 1:
            tX_train = tx_train
            tX_test = tx_test
        else:
            # Build the polynomials functions
            tX_train = build_poly(tx_train, degree)
            tX_test = build_poly(tx_test, degree) 

        mats_test.append(tX_test)
        mats_train.append(tX_train)
        
    return mats_train, pred_train, mats_test, pred_test

def find_min(rmse_te, lambdas, degrees):
    print("Min for rmse_te: %f"%(np.min(rmse_te)))
    x, y = np.where(rmse_te == np.min(rmse_te))
    ilamb_star = x[0]
    ideg_star = y[0]
    print("test = %f"%(rmse_te[ilamb_star,ideg_star]))
    
    return lambdas[ilamb_star], int(degrees[ideg_star])