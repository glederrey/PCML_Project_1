# -*- coding: utf-8 -*-
"""
Costs
"""

import numpy as np

def compute_loss(y, tx, w, loss_str):
	if loss_str == "MSE":
		return compute_MSE(y, tx, w)
	elif loss_str == "RMSE":
		return compute_RMSE(y, tx, w)
	elif loss_str == "MAE":
		return compute_MAE(y, tx, w)
	else:
		raise ValueError("The loss provided \"%s\" does not exist. You have the choice between MSE, RMSE or MAE"%loss_str)

def compute_MSE(y, tx, w):
	"""Compute the loss using MSE"""

    e = y - np.dot(tx, w)
    N = len(y)
    
    L = 1/(2*N)*np.dot(np.transpose(e),e)
    return L

def compute_RMSE(y, tx, w):
	"""Compute the loss using RMSE"""
	"""Corresponds to sqrt(2*MSE)"""
	
	return np.sqrt(2*compute_MSE(y, tx, w))

def compute_MAE(y, tx, w):
	"""Compute the loss using MAE"""

    e = y - np.dot(tx, w)
    N = len(y)
    
    L = 1/(N)*np.sum(np.abs(e))
    return L 	
