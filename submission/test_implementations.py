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
    x_train, y_train, x_test, y_test = split_data(tX, y, ratio)

    # Test the GD
    #test_GD(x_train, y_train, x_test, y_test)
    
    # Test the SGD
    #test_SGD(x_train, y_train, x_test, y_test)
    
    # Test Least Square
    #test_Least_Square(x_train, y_train, x_test, y_test)
    
    # Test Ridge Regression

def test_GD(x_train, y_train, x_test, y_test):
  
    print("TEST OF THE GD")
    gamma = 0.1
    max_iters = 300
    initial_w = np.ones(len(x_train[0]))
    
    w, loss = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma)
    
    prediction(y_test, x_test, w)
    
def test_SGD(x_train, y_train, x_test, y_test):
    
    print("TEST OF THE SGD")
    gamma = 1e-3
    max_iters = 300
    initial_w = np.zeros(len(x_train[0]))
    
    w, loss = least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma)
    
    prediction(y_test, x_test, w)
    
def test_Least_Square(x_train, y_train, x_test, y_test):
    print("TEST OF THE LEAST SQUARE")
    
    w, loss = least_squares(y_train, x_train)
    
    prediction(y_test, x_test, w)    

if __name__ == "__main__":
    main()