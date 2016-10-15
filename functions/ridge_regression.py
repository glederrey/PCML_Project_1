# -*- coding: utf-8 -*-
"""
Ridge Regression
"""

import numpy as np
from functions.costs import *
import matplotlib.pyplot as plt
from IPython import display

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""

    # Compute optimal weights
    xx = np.dot(np.transpose(tx),tx)
    
    bxx = xx + lamb*np.identity(len(xx))
        
    xy = np.dot(np.transpose(tx),y)
    w_star = np.linalg.solve(bxx, xy)
    
    loss = compute_cost(y, tx, w_star, 'RMSE')
    
    return loss, w_star
    
def test_ridge_regression(y_train, x_train, y_test, x_test):
    """ Test the ridge regression and returns all losses and all weights"""
    lambdas = np.logspace(-10, 10, 200)
    ws = []
    losses_train = []
    losses_test = []
    lb = []

    plt.title("Loss in function of lambda.")
    plt.xlabel("Lambda")
    plt.ylabel("Loss") 
    
    first_iter = True
    
    for lambda_ in lambdas:
        loss, w = ridge_regression(y_train, x_train, lambda_)
        ws.append(w)
        losses_train.append(loss)
        lb.append(lambda_)
        
        test_loss = compute_cost(y_test, x_test, w, 'RMSE')
        
        losses_test.append(test_loss)
        
        # Plot the graph of the loss   
        if first_iter:   
            plt.loglog(lb, losses_train, '-*b', label='train')
            plt.loglog(lb, losses_test, '-*r', label='test')
            plt.legend(loc=3)
        else:
            plt.loglog(lb, losses_train, '-*b')
            plt.loglog(lb, losses_test, '-*r')        
        plt.title("Loss in function of lambda.\ntrain loss = %f\ntest loss = %s"%(loss, test_loss))         
        display.display(plt.gcf())        
        display.clear_output(wait=True)
        
        first_iter = False  
            
