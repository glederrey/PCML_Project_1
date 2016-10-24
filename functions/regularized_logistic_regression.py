# -*- coding: utf-8 -*-
"""
Regularized Logistic Regression using Gradient Descent
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from functions.logistic_regression import *
from functions.helpers import *
from functions.costs import *

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w) + lambda_*np.linalg.norm(w)**2
    grad = calculate_gradient(y, tx, w) + 2*lambda_*w
    return loss, grad

def learning_by_penalized_gradient(y, tx, w, alpha, lambda_):
    """
        Do one step of gradient descent, using the penalized logistic regression.
        Return the loss and updated w.
        """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w-alpha*gradient
    return loss, w
    
def regularized_logistic_regression(y, tx, gamma, lambda_, max_iters, draw=True, verbose=True):
    """
        Use the logistic regression with Gradient Descent method.
    """
    # define initial_w
    initial_w = np.ones(len(tx[0]))

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    iterations = []
    
    if draw:
        plt.title("Loss in function of epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss") 
    
    loss_str = 'MSE'
    
    last_loss = 0
    
    for n_iter in range(max_iters):
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        if draw:
            plt.semilogy(iterations, losses, '-*b') 
            plt.title("Loss in function of epochs.\nLast loss = %10.3e"%loss)
            display.display(plt.gcf())        
            display.clear_output(wait=True)
        elif verbose:
            if n_iter % 100 == 0:
                print("  Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))
                last_loss = loss
              
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 1e-8:
            return losses, ws

    if verbose:
        print("  Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))     
   
    return np.array(losses), np.array(ws)
    
def cross_validation(y, tx, lambdas, degrees, gamma, max_iter, k_fold, verb = False, seed = 1):
    """
        K-fold cross validation for the Logistic Regression
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
                loss = calculate_cv(y, tx, k_indices, k, lamb, deg, gamma, max_iter)
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
    
def calculate_cv(y, tx, k_indices, k, lamb, degree, gamma, max_iter):
    # Values for the test 
    tx_test = tx[k_indices[k]]    
    y_test = y[k_indices[k]]   
    
    # Get all the indices that are not in the test data
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
    
    # Apply the Regularized Logistic Regression
    losses, ws = regularized_logistic_regression(y_train, tX_train, gamma, lamb, max_iter, False, False) 
    
    w_star, _ = get_best_model(losses, ws)
    
    # Return the RMSE on the test data
    return compute_cost(y_test, tX_test, w_star, 'RMSE')

def find_min(rmse_te, lambdas, degrees):
    print("Min for rmse_te: %f"%(np.min(rmse_te)))
    x, y = np.where(rmse_te == np.min(rmse_te))
    ilamb_star = x[0]
    ideg_star = y[0]
    print("test = %f"%(rmse_te[ilamb_star,ideg_star]))
    
    return lambdas[ilamb_star], int(degrees[ideg_star])