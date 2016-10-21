# -*- coding: utf-8 -*-
"""
Logic Regression
"""

import numpy as np
from functions.costs import *
import matplotlib.pyplot as plt
from IPython import display
from functions.least_squares_GD import *

def logistic_regression(y, tx, gamma, max_iters):
    """
        Use the logistic regression with Newton method.
    """
    # define initial_w
    initial_w = np.ones(len(tx[0]))

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
        # Compute the gradient and the loss
        loss = calculate_loss(y, tx, w)
        grad = calculate_gradient(y, tx, w)
        
        """ Newton
        # compute the Hessian
        hessian = calculate_hessian(y, tx, w)
        
        # Update w by Newton step
        inv_hess_grad = np.linalg.solve(hessian, grad)

        w = w - gamma * np.squeeze(inv_hess_grad)
        """
        
        # Update by GD step
        w = w - gamma * np.squeeze(grad)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        #print("Gradient Descent({bi}/{ti}): loss={l}, grad={grad}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, grad=np.linalg.norm(grad)))
        plt.plot(iterations, losses, '-*b') 
        plt.title("Loss in function of epochs.\nLast loss = %f"%loss)
        display.display(plt.gcf())        
        display.clear_output(wait=True)      
              
        if n_iter > 1 and np.abs(losses[-1]-losses[-2]) < 10**-10:
            return losses, ws

    return losses, ws          
    
def sigmoid(t):
    """apply sigmoid function on t.
    Apply it to each row of the vecotr given in input
    """
    
    sig = np.zeros((t.shape[0],1))
    for i in range(t.shape[0]):
        #y[i] = math.exp(t[i]) / (1 + math.exp(t[i]))
        sig[i] = 1/(1+np.exp(-t[i]))

    return sig
    
def calculate_loss(y, tx, w):
    """
        Compute the loss using the Log-Likelihood
    """
    
    L = 0
    for i in range(tx.shape[0]):
        exp = np.exp(np.dot(tx[i], w))
        log_n = np.log(1+exp)
        yxw = y[i] * np.dot(np.transpose(tx[i]), w)
        L += log_n - yxw
        
    return L
    
def calculate_gradient(y, tx, w):
    """
        Compute the Gradient using the sigmoid
    """

    sig = sigmoid(np.dot(tx,w))
    y = y[:, np.newaxis] # Problem of memory otherwise
    diff = np.subtract(sig, y)
    grad = np.dot(np.transpose(tx), diff)
    
    return grad
    
def calculate_hessian(y, tx, w):
    """
        Compute the Hessian using the sigmoid
    """
    
    sig = sigmoid(np.dot(tx,w))
    one = np.ones((len(sig),1))
    diff = np.subtract(one,sig)
    s_diag = np.squeeze(np.multiply(sig, diff))
    ttx = np.transpose(tx)
    ttx_s = np.zeros((len(ttx), len(s_diag)))

    for i in range(len(s_diag)):
        for j in range(len(ttx)):
            ttx_s[j,i] += s_diag[i]*ttx[j,i]
            
    h = np.dot(ttx_s, tx)
    return h 
    
    

