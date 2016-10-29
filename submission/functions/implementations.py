# -*- coding: utf-8 -*-
"""
    This file contains the 6 mandatory functions for the project 1 in PCML.
"""

import numpy as np
from helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
        Use the Gradient Descent method to find the best weights
        
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
    losses = []
    w = initial_w
    iterations = []
    
    last_loss = 0    
    
    
    for n_iter in range(max_iters):
        # Compute the gradient and the loss (See helpers.py for the functions)
        loss = compute_cost(y, tx, w)
        grad = compute_gradient(y, tx, w)
                
        # Update w by gradient
        w = w - gamma*grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        
        if n_iter % 100 == 0:
            print("  Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))
            last_loss = loss        
          
        # Stopping criteria for the convergence
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 10**-8:
            break
            
    print("  Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))  
    # Get the latest loss and weights
    return ws[-1], losses[-1]
            
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
        Use the Stochastic Gradient Descent (batch size 1) method to find the best weights
        
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
       
    # Define a batch size of 1 for the submission
    batch_size = int(1)
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    iterations = []
    
    last_loss = 0    
    
    for n_iter in range(max_iters):
        # Compute the stochastic gradient and the loss (See helpers.py for the functions)
        loss = compute_cost(y, tx, w)
        grad = compute_stoch_gradient(y, tx, w, batch_size, 100)
                
        # Update w by gradient
        w = w - gamma*grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        
        if n_iter % 100 == 0:
            print("  Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))
            last_loss = loss        
          
        # Stopping criteria for the convergence
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 10**-8:
            break
            
    print("  Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))  
    # Get the latest loss and weights
    return ws[-1], losses[-1]   

def least_squares(y, tx):
    """
        Use the Least Square method to find the best weights
        
        INPUT:
            y           - Predictions
            tx          - Samples
            
        OUTPUT:
            w           - Best weights
            loss        - Minimum loss
    """    
    # Compute optimal weights
    xx = np.dot(np.transpose(tx),tx)        
    xy = np.dot(np.transpose(tx),y)
    w_star = np.linalg.solve(xx, xy)
    
    loss = compute_RMSE(y, tx, w_star)
    
    return w_star, loss
    
def ridge_regression(y, tx, lambda_):
    """
        Use the Ridge Regression method to find the best weights
        
        INPUT:
            y           - Predictions
            tx          - Samples
            
        OUTPUT:
            w           - Best weights
            loss        - Minimum loss
    """    

    # Compute optimal weights
    xx = np.dot(np.transpose(tx),tx)
    # Add the lambda on the diagonal
    bxx = xx + lambda_*np.identity(len(xx))     
    xy = np.dot(np.transpose(tx),y)  
    w_star = np.linalg.solve(bxx, xy)
        
    loss = compute_RMSE(y, tx, w_star)
    
    return w_star, loss
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
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
    losses = []
    w = initial_w
    iterations = [] 
       
    last_loss = 0
    
    for n_iter in range(max_iters):
        # Gradient descent method
        loss = calculate_loss_logit(y, tx, w)
        grad = calculate_gradient_logit(y, tx, w)
        w = w - gamma*grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        if n_iter % 100 == 0:
            print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))
            last_loss = loss
            
        # Stopping criteria for the convergence
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 1e-8:
            break

    print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))     
    
    return ws[-1], losses[-1]
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
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
    losses = []
    w = initial_w
    iterations = [] 
       
    last_loss = 0
    
    for n_iter in range(max_iters):
        # Gradient descent method
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        if n_iter % 100 == 0:
            print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))
            last_loss = loss
            
        # Stopping criteria for the convergence
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 1e-8:
            break

    print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))     
    
    return ws[-1], losses[-1]