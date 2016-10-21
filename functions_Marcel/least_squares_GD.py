# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np


def compute_loss(y, tx, w):
    error = y - np.dot(tx, w)
    loss = 1/(2*np.size(y))*np.dot(np.transpose(error), error)
    return loss

def compute_gradientLS(y, tx, w):
    error = y.ravel() - np.dot(tx, w)
    grad = -1/y.size* np.dot(np.transpose(tx), error)
    return grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma, Print = 0):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        grad = compute_gradientLS(y, tx, w)
        w = w - gamma*grad
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        ws.append(w)
        losses.append(loss)
        if Print:   
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return losses, w
