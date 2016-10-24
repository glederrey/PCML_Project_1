# -*- coding: utf-8 -*-
"""logistic regression using gradient descent
"""
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return (np.sum(np.log(1+np.exp(np.dot(tx,w)))) - np.dot(y.transpose(),np.dot(tx,w)))


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
    
def logistic_regression(y, tx, gamma, max_iters, draw=True):
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
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        if draw:
            plt.semilogy(iterations, losses, '-*b') 
            plt.title("Loss in function of epochs.\nLast loss = %10.3e"%loss)
            display.display(plt.gcf())        
            display.clear_output(wait=True)
        else:
            if n_iter % 100 == 0:
                print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))
                last_loss = loss
              
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 1e-8:
            return losses, ws

    if not draw:
        print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss-last_loss)))     
    
    return losses, ws