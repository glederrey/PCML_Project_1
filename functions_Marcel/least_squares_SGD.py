# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""

import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
        Generate a minibatch iterator for a dataset.
        Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
        Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
        Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
        Example of use :
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
        """
    data_size = len(y)
    
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_loss(y, tx, w):
    error = y.ravel() - np.dot(tx, w)
    loss = 1/(2*np.size(y))*np.dot(np.transpose(error), error)
    return loss

def compute_gradient(y, tx, w):
    error = y.ravel() - np.dot(tx, w)
    grad = -1/y.size* np.dot(np.transpose(tx), error)
    return grad

def compute_stoch_gradient(y, tx, w, batch_size):
    stoch_grad = np.zeros(len(tx[0]))
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        stoch_grad = stoch_grad + compute_gradient(minibatch_y, minibatch_tx, w)
    return 1/batch_size* stoch_grad

def least_squares_SGD(
y, tx, initial_w, batch_size, max_epochs, gamma, Print = 0):
        ws = [initial_w]
        losses = []
        w = initial_w
        for n_iter in range(max_epochs):         
            loss = compute_loss(y, tx, w)
            grad = compute_stoch_gradient(y, tx, w, batch_size)
            w = w - gamma*grad
            # store w and loss
            ws.append(np.copy(w))
            losses.append(loss)
            if Print:
                print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_epochs- 1, l=loss, w0=w[0], w1=w[1]))
        return losses, w
