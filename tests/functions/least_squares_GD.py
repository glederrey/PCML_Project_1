# -*- coding: utf-8 -*-
"""
Least Squares with Gradient Descent
"""

import numpy as np
from functions.costs import *
import matplotlib.pyplot as plt
from IPython import display
from functions.helpers import *

def least_squares_GD(y, tx, max_iters, gamma):
    #initial_w = least_squares(y, tx)
    initial_w = np.ones(len(tx[0]))

    return gradient_descent(y, tx, initial_w, max_iters, gamma, 'MAE')

def compute_gradient(y, tx, w, loss_str):
    """Compute the gradient."""

    e = y - np.dot(tx, w)
    N = len(y)

    if loss_str == "MSE":
        return -1/N*np.dot(np.transpose(tx), e)
    elif loss_str == "MAE":
        val = 0
        for i in range(len(e)):
            # np sign puts 0 when e[i]=0
            val = val + np.sign(e[i])

        return 1/N*val*w
    else:
        raise ValueError("The loss provided \"%s\" does not exist. You have the choice between MSE or MAE"%loss_str)        
        
def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_str):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    iterations = []
    
    last_loss = 0
    
    for n_iter in range(max_iters):
        # Compute the gradient and the loss
        loss = compute_cost(y, tx, w, loss_str)
        grad = compute_gradient(y, tx, w, loss_str)
        
        # Update w by gradient
        w = w - gamma*grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        
        if n_iter % 100 == 0:
            print("  Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))
            last_loss = loss        
          
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 10**-8:
            print("  Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))     

            return losses, ws
            
    print("  Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))     

    return losses, ws
