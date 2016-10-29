# -*- coding: utf-8 -*-
"""
    This file test the 6 mandatory functions defined in the file implementations.py
"""

from implementations import *
from helpers import *

def main():
    print("LOADING THE DATA")
    # Prepare the data
    y, tX, ids = load_csv_data('train.csv')
    
    tX, _, _ = standardize(tX)

    ratio = 0.8
    x_train, y_train, x_test, y_test = split_data(tX, y, ratio, seed=1)

    # Test the GD
    #test_GD(x_train, y_train, x_test, y_test)
    
    # Test the SGD
    #test_SGD(x_train, y_train, x_test, y_test)
    
    # Test Least Square
    #test_Least_Square(x_train, y_train, x_test, y_test)
    
    # Test Ridge Regression
    #test_Ridge_Regression(x_train, y_train, x_test, y_test)
    
    # Test Logistic Regression
    #test_Logistic_Regression(x_train, y_train, x_test, y_test)

    # Test Regularized Logistic Regression    
    test_Reg_Logistic_Regression(x_train, y_train, x_test, y_test)

def test_GD(x_train, y_train, x_test, y_test):
  
    print("TEST OF THE GD")
    gamma = 0.1
    max_iters = 300
    initial_w = np.ones(len(x_train[0]))
    
    w, loss = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma)
    
    prediction(y_test, x_test, w)
    print("")    
    
def test_SGD(x_train, y_train, x_test, y_test):
    
    print("TEST OF THE SGD")
    gamma = 1e-3
    max_iters = 300
    initial_w = np.zeros(len(x_train[0]))
    
    w, loss = least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma)
    
    prediction(y_test, x_test, w)
    print("")
     
def test_Least_Square(x_train, y_train, x_test, y_test):
    print("TEST OF THE LEAST SQUARE")
    
    w, loss = least_squares(y_train, x_train)
    
    prediction(y_test, x_test, w)
    print("")

def test_Ridge_Regression(x_train, y_train, x_test, y_test):
    print("TEST OF THE RIDGE REGRESSION")
    
    lambda_ = 1e-2
    
    w, loss = ridge_regression(y_train, x_train, lambda_)
    
    prediction(y_test, x_test, w)
    print("")
    
def test_Logistic_Regression(x_train, y_train, x_test, y_test):
    
    print("TEST OF THE LOGISTIC REGRESSION")
    
    # First thing we need to do is to prepare the data to use them with the logit.
    # For the logit, the predictions are 0 or 1. Therefore, we need to transform the 
    # predictions y. We also transform the values for x such that they all are between 
    # 0 and 1. It helps the logistic regression to work better.
    x_train, y_train = prepare_logit(x_train, y_train)    
    x_test, y_test = prepare_logit(x_test, y_test)   

    gamma = 1e-6
    max_iters = 2000
    initial_w = np.ones(len(x_train[0]))
   
    w, loss = logistic_regression(y_train, x_train, initial_w, max_iters, gamma)
    
    prediction_logit(y_test, x_test, w)
    print("") 

def test_Reg_Logistic_Regression(x_train, y_train, x_test, y_test):
    
    print("TEST OF THE REGULARIZED LOGISTIC REGRESSION")
    
    # First thing we need to do is to prepare the data to use them with the logit.
    # For the logit, the predictions are 0 or 1. Therefore, we need to transform the 
    # predictions y. We also transform the values for x such that they all are between 
    # 0 and 1. It helps the logistic regression to work better.
    x_train, y_train = prepare_logit(x_train, y_train)    
    x_test, y_test = prepare_logit(x_test, y_test)   

    gamma = 1e-6
    max_iters = 2000
    lambda_ = 2
    initial_w = np.ones(len(x_train[0]))
   
    w, loss = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)
    
    prediction_logit(y_test, x_test, w)
    print("")    

if __name__ == "__main__":
    main()