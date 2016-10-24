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
    
    #plt.figure()
    plt.title("Loss in function of epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    
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
        #print("Gradient Descent({bi}/{ti}): loss={l}, grad={grad}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, grad=np.linalg.norm(grad)))
        plt.semilogy(iterations, losses, '-*b') 
        plt.title("Loss in function of epochs.\nLast loss = %f"%loss)
        display.display(plt.gcf())        
        display.clear_output(wait=True)      
              
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 10**-10:
            return losses, ws

    return losses, ws
