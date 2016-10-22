# -*- coding: utf-8 -*-
"""logistic regression using gradient descent
"""
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from functions.least_squares_GD import *
import time

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.exp(-t))


def calculate_loss(y, tx, w):
    return (np.sum(1+np.exp(np.dot(tx,w))) -np.dot(y.transpose(),np.dot(tx,w)))


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return (np.dot(tx.transpose(),sigmoid(np.dot(tx,w))-y))


def learning_by_gradient_descent(y, tx, w, alpha):
    """
        Do one step of gradient descen using logistic regression.
        Return the loss and the updated w.
        """
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    w = w-alpha*grad
    return loss, w
    
def logistic_regression(y, tx, gamma, max_iters):
    """
        Use the logistic regression with Gradient Descent method.
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
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)

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
