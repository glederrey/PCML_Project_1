# -*- coding: utf-8 -*-
"""
Least Squares with Stochastic Gradient Descent
"""

import numpy as np
from functions.costs import *
import matplotlib.pyplot as plt
from IPython import display
from functions.helpers import *
from functions.least_squares_GD import *

def least_squares_SGD(y, tx, max_iters, gamma):
    #initial_w = least_squares(y, tx)
    initial_w = np.ones(len(tx[0]))
    
    n = len(y)
    
    batch_size = int(500)

    return stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, 'MAE')

def compute_stoch_gradient(y, tx, w, batch_size, loss_str):
    """Compute a stochastic gradient for batch data."""

    stoch_grad = np.zeros(len(tx[0]))
 
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        stoch_grad = stoch_grad + compute_gradient(minibatch_y, minibatch_tx, w, loss_str)
        
    return 1/batch_size * stoch_grad


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma, loss_str):
    """Stochastic gradient descent algorithm."""
 
    ws = [initial_w]
    losses = []
    w = initial_w
    iterations = []    
    
    #plt.figure()
    plt.title("Loss in function of epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")    
    
    for n_iter in range(max_epochs):
		# Compute gradient and loss
        
        loss = compute_cost(y, tx, w, loss_str)
        grad = compute_stoch_gradient(y, tx, w, batch_size, loss_str)        
        
		# Update w by gradient
        w = w - gamma * grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        
        # Plot the graph of the loss        
        plt.semilogy(iterations, losses, '-*b')
        plt.title("Loss in function of epochs.\nLast loss = %f"%loss)         
        display.display(plt.gcf())        
        display.clear_output(wait=True)
        
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 10**-10:
            return losses, ws              

    return losses, ws 
