# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np

def compute_cost(y, tx, w, loss_str):
    # Compute the error
    e = y - tx.dot(w)

    if loss_str == "MSE":
        return compute_MSE(e)
    elif loss_str == "RMSE":
        return compute_RMSE(e)
    elif loss_str == "MAE":
        return compute_MAE(e)
    else:
        raise ValueError("The loss provided \"%s\" does not exist. You have the choice between MSE, RMSE or MAE"%loss_str)

def compute_MSE(e):
    """Compute the loss using MSE"""

    return 1/2*np.mean(e**2)

def compute_RMSE(e):
    """Compute the loss using RMSE"""
    """Corresponds to sqrt(2*MSE)"""
    
    return np.sqrt(2*compute_MSE(e))

def compute_MAE(e):
    """Compute the loss using MAE"""

    return np.mean(np.abs(e))
