# -*- coding: utf-8 -*-
"""logistic regression using gradient descent
"""
import numpy as np
from helpers import de_standardize
from plots import visualization

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

def reg_logistic_regression(y, tx, gamma,lambda_, Print = 0, Visualization = 0, max_iter = 10000):
    threshold = 1e-8
    losses = []
    num_samples = len(y)
    w = np.zeros((tx.shape[1], 1))
    
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if Print:
            if iter % 1000 == 0:
                print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    if Print:
        print("The loss={l}".format(l=calculate_loss(y, tx, w)))
    return loss, w
