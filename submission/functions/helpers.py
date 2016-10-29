# -*- coding: utf-8 -*-
"""
    This file contains all additional functions that are used in the mandatory functions in the file implementations.py.
"""

import numpy as np 

""" ----------- FUNCTIONS FOR GRADIENT DESCENT ----------- """  
 
def compute_gradient(y, tx, w):
    """
        Compute the gradient for the Gradient Descent method
        
        INPUT:
            y           - Predictions vector
            tx          - Samples
            w           - Weights
            
        OUTPUT:
            Return the gradient for the given input
    """

    e = y - np.dot(tx, w)
    N = float(len(y))
    
    return -1./N*np.dot(np.transpose(tx), e)
    
""" ----------- FUNCTIONS FOR STOCHASTIC GRADIENT DESCENT ----------- """ 

def compute_stoch_gradient(y, tx, w, batch_size, max_iter):
    """Compute a stochastic gradient for batch data."""

    stoch_grad = np.zeros(len(tx[0]))
 
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iter):
        stoch_grad = stoch_grad + compute_gradient(minibatch_y, minibatch_tx, w)
        
    return 1/float(batch_size) * stoch_grad
    
""" ----------- FUNCTIONS FOR LOGISTIC REGRESSION ----------- """ 
def sigmoid(t):
    """
        Apply sigmoid function on t. 
        We used the version 1/(1+exp(-t)) to avoid overflows.
    """
    return 1./(1.+np.exp(-t))
    
def calculate_loss_logit(y, tx, w):
    """
        Compute the cost by negative log likelihood for the Logistic Regression (Logit)
    """
    return np.sum(np.log(1+np.exp(np.dot(tx,w)))) - np.dot(y.transpose(),np.dot(tx,w))
    
def calculate_gradient_logit(y, tx, w):
    """
        Compute the gradient of loss for the Logistic Regression (Logit)
    """
    return (np.dot(tx.transpose(),sigmoid(np.dot(tx,w))-y))
    
def prepare_logit(x, y):
    """
        Standardize the original data set. Using the following feature scaling:
            X = (X - Xmin) / (Xmax - Xmin)
        The predictions are change such that all the values equal to -1 becomes 0.
    """
    x  = ((x.T - x.min(1)) / (x.max(1) - x.min(1))).T 
    # Add the first column of ones.
    tx = np.hstack((np.ones((x.shape[0],1)), x))
	
    y[y==-1] = 0
    return tx, y
   
""" ----------- FUNCTIONS FOR REGULARIZED LOGISTIC REGRESSION ----------- """
   
def penalized_logistic_regression(y, tx, w, lambda_):
    """
        Return the Loss and the gradient of the regularized logistic regression
    """
    loss = calculate_loss_logit(y, tx, w) + lambda_*np.linalg.norm(w)**2
    grad = calculate_gradient_logit(y, tx, w) + 2*lambda_*w
    return loss, grad

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
        Do one step of gradient descent, using the penalized logistic regression.
        Return the loss and updated w.
        """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma*gradient
    return loss, w
 
""" ----------- COSTS ----------- """

def compute_cost(y, tx, w):
    """
        Compute the MSE cost.
        
        INPUT:
            y           - Predictions vector
            tx          - Samples
            w           - Weights
            
        OUTPUT:
            cost        - Double value for the costs seen in the course.
    """
    # Compute the error
    e = y - tx.dot(w)

    # Compute the cost
    return 1./2*np.mean(e**2)

def compute_RMSE(y, tx, w):
    """
        Compute the RMSE cost given the the inputs for the cost.
    """
    
    return np.sqrt(2*compute_cost(y, tx, w))
    
""" ----------- FUNCTIONS TO TEST THE PREDICTIONS ----------- """
    
def prediction(y, tX, w_star):

    pred = np.dot(tX, w_star)

    pred[pred>0] = 1
    pred[pred<=0] = -1
    
    right = np.sum(pred == y)
    wrong = len(pred)-right
            
    print("Good prediction: %i/%i (%.3f%%)\nWrong prediction: %i/%i (%.3f%%)"%
          (right, len(y), 100.0*float(right)/float(len(y)), 
          wrong,  len(y), 100.0*float(wrong)/float(len(y))))
          
def prediction_logit(y, tX, w_star):
    """
        For the Logistic Regression, we need a different function because
        the predictions are 1 or 0. So, instead of checking if the float value
        is positive or negative, we need to check if it is bigger or smaller than 0.5
    """

    pred = np.dot(tX, w_star)

    pred[pred>0.5] = 1
    pred[pred<=0.5] = -1
    
    right = np.sum(pred == y)
    wrong = len(pred)-right
            
    print("Good prediction: %i/%i (%.3f%%)\nWrong prediction: %i/%i (%.3f%%)"%
          (right, len(y), 100.0*float(right)/float(len(y)), 
          wrong,  len(y), 100.0*float(wrong)/float(len(y))))
    
""" ----------- FUNCTIONS FOR CROSS-VALIDATION ----------- """
    
def build_k_indices(y, k_fold, seed):
    """
        Build k-indices for the Cross-Validation
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def build_poly(x, degree):
    """
        Build the polynomial basis functions seen during the lecture
        
        INPUT:
            x       - Sample matrix
            degree  - degree for the polynomial basis
            
        OUTPUT:
            matrix with the polynomial basis of degree "degree"
    """
    # We test if the degree is higher than 1 because if the degree is 1,
    # we will just add columns of 1 which is useless.
    if degree > 1:
        n_x = len(x)
        nbr_param = len(x[0])
        mat = np.zeros((n_x, (degree+1)*nbr_param))
        
        # For each feature, we add degree+1 columns 
        for j in range(nbr_param):
            for k in range(degree+1):
                mat[:, j*(degree+1)+k] = x[:,j]**k
                
        return mat
    else:
        # If it's degree 1, we just return the matrix.
        return x
    
def ct_poly(x, degree):
    """
        Build the polynomial basis functions seen during the lecture
        and in addition, we add the pairs of cross-terms at the end
        
        INPUT:
            x       - Sample matrix
            degree  - degree for the polynomial basis
            
        OUTPUT:
            matrix with the polynomial basis of degree "degree" and in addition all
            the pairs of cross terms are added to the matrix.
    """
    n_x = len(x)
    
    nbr_param = len(x[0])    
    
    # nbr of cross terms (it corresponds to pick 2 in the number of feature (binomial))
    nbr_ct = int(nbr_param*(nbr_param-1)/2)

    # We test if the degree is higher than 1 because if the degree is 1,
    # we will just add columns of 1 which is useless for the polynomial 
    # basis matrix.
    if degree > 1:
        mat = np.zeros((n_x, (degree+1)*nbr_param + nbr_ct))
        
        # The first (degree+1)*nbr_param columns are the polynomial basis matrix
        for j in range(nbr_param):
            for k in range(degree+1):
                mat[:, j*(degree+1)+k] = x[:,j]**k
        
        # The rest of the columns are all the cross terms
        idx = (degree+1)*nbr_param
        for l in range(nbr_param):
            for m in range(l+1, nbr_param):
                mat[:, idx] = x[:,l]*x[:,m]
                idx += 1
                
    elif degree==1:
        mat = np.zeros((n_x, nbr_param + nbr_ct))
        
        # The first nbr_param columns is the matrix itself
        mat[:, :nbr_param] = x
        
        # The rest of the columns are all the cross terms
        idx = nbr_param
        for l in range(nbr_param):
            for m in range(l+1, nbr_param):
                mat[:, idx] = x[:,l]*x[:,m]
                idx += 1
                
    return mat  
    
""" ----------- FUNCTIONS USED FOR THE LOGISTIC REGRESSION ----------- """ 

def prepare_log_reg(x, y):
    """
    Standardize the original data set. Using feature scaling:
                X = (X - Xmin) / (Xmax - Xmin)
                
    The predictions of -1 are set to 0.
                
    INPUT:
        x   - Samples matrix
        y   - Prediction vector
        
    OUTPUT:
        Rescaled version of the Samples matrix and the Prediction vector.
    """
    x  = ((x.T - x.min(1)) / (x.max(1) - x.min(1))).T 
    # Add the column of ones at the beginnin
    tx = np.hstack((np.ones((x.shape[0],1)), x))
	# Replace all the -1 by 0
    y[y==-1] = 0
    return tx, y
    
""" ----------- HELPERS FUNCTIONS (Given by the professor) ----------- """

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def standardize(x, mean_x=None, std_x=None):
    """
        Standardize the original data set. (Given by the professor)
    """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x
    
def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
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
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

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

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred
    
def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, id
    
def get_best_model(gradient_losses, gradient_ws):
    min_loss = gradient_losses[0]
    best_model = gradient_ws[0]
    
    for i in range(len(gradient_losses)-1):
        if gradient_losses[i+1] < min_loss:
            min_loss = gradient_losses[i+1]
            best_model = gradient_ws[i+1]
            
    return best_model, min_loss 
    
def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    
    np.random.seed(seed)    
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
    