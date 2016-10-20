# -*- coding: utf-8 -*-
"""Split function"""
import numpy as np

def split_data(x, y, ratio):
    """split the dataset based on the split ratio."""
    # set seed
    #np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    
    #raise NotImplementedError
    
    n = len(x)
    if len(y) != n:
        raise ValueError("Vector x and y have a different size")
        
    n_train = int(ratio*n)
    train_ind = np.random.choice(n, n_train, replace=False)
        
    index = np.arange(n)
    
    mask = np.in1d(index, train_ind)
    
    test_ind = np.random.permutation(index[~mask])
    
    x_train = x[train_ind]
    y_train = y[train_ind]
    
    x_test = x[test_ind]
    y_test = y[test_ind]
    
    return x_train, y_train, x_test, y_test 
    
def split_non_random(x, y, ratio):
    """ Split the dataset based on the split ratio.
        In this function, we just take the first ratio*n entries 
        as the training set and the rest as the testing test.
        
        It's mostly used when trying to clean the data to remove 
        the random part. """
        
    n = len(x)
    if len(y) != n:
        raise ValueError("Vector x and y have a different size")
        
        
    n_train = int(ratio*n)
    x_train = np.array(x[:n_train], dtype=np.float64)
    y_train = np.array(y[:n_train], dtype=np.float64)
    
    x_test = np.array(x[n_train:], dtype=np.float64)
    y_test = np.array(y[n_train:], dtype=np.float64) 
    
    return x_train, y_train, x_test, y_test       
        
