# -*- coding: utf-8 -*-
"""
Gradient Descent
"""
import numpy as np
import costs

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
    for n_iter in range(max_iters):
		# Compute the gradient and the loss
		loss = compute_cost(y, tx, w, loss_str)
        grad = compute_gradient(y, tx, w, loss_str)

		# Update w by gradient

		w = w + gamma*grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
