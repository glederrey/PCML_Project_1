# -*- coding: utf-8 -*-
"""
Regularized Logistic Regression using Gradient Descent
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from functions.least_squares_GD import *

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.exp(-t))


def calculate_loss(y, tx, w):
    return (np.sum(1+np.exp(np.dot(tx,w))) -np.dot(y.transpose(),np.dot(tx,w)))


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return (np.dot(tx.transpose(),sigmoid(np.dot(tx,w))-y))


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
    
def regularized_logistic_regression(y, tx, gamma, lambda_, max_iters):
    """
        Use the regularized logistic regression with Gradient Descent method.
    """
    # define initial_w
    initial_w = np.zeros(len(tx[0]))

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    iterations = []
    
    #plt.figure()
    plt.title("Loss in function of epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss") 
    
    loss_str = 'MSE'
    
    for n_iter in range(max_iters):
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        #print("Gradient Descent({bi}/{ti}): loss={l}, grad={grad}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, grad=np.linalg.norm(grad)))
        plt.semilogy(iterations, losses, '-*b') 
        plt.title("Loss in function of epochs.\nLast loss = %10.3e"%loss)
        display.display(plt.gcf())        
        display.clear_output(wait=True)      
              
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 1e-8:
            return losses, ws

    return losses, ws
