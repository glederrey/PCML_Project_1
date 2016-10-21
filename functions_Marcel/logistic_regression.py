# -*- coding: utf-8 -*-
"""logistic regression using gradient descent
"""
import numpy as np
from plots import visualization

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

def logistic_regression(y, tx,  gamma , Print = 0, max_iter = 10000):
    threshold = 1e-8
    losses = []
    w = np.zeros((tx.shape[1], 1))
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if Print:
            if iter % 1000 == 0:
                print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    if Print:
        print("The loss={l}".format(l=calculate_loss(y, tx, w)))
    return loss, w
