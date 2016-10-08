# -*- coding: utf-8 -*-
"""
Stochastic Gradient Descent
"""
from helpers import batch_iter

def compute_stoch_gradient(y, tx, w, loss_str):
    """Compute a stochastic gradient for batch data."""

    stoch_grad = np.array([0,0])
 
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        stoch_grad = stoch_grad + compute_gradient(minibatch_y, minibatch_tx, w, loss_str)
        
    return 1/batch_size * stoch_grad


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma, loss_str):
    """Stochastic gradient descent algorithm."""
 
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_epochs):
		# Compute gradient and loss
        
        loss = compute_loss(y, tx, w, loss_str)
        grad = compute_stoch_gradient(y, tx, w, batch_size, loss_str)
        
		# Update w by gradient
        
        w = w - gamma * grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws    
