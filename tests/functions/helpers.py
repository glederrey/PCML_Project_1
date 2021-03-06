# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import csv

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x
    
def std(x):
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    return x

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
    
def get_best_model(gradient_losses, gradient_ws):
    min_loss = gradient_losses[0]
    best_model = gradient_ws[0]
    
    for i in range(len(gradient_losses)-1):
        if gradient_losses[i+1] < min_loss:
            min_loss = gradient_losses[i+1]
            best_model = gradient_ws[i+1]
            
    return best_model, min_loss 
    
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


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred
    
def predict_labels_logit(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
def prediction(y, tX, w_star):

    pred = np.dot(tX, w_star)

    pred[pred>0] = 1
    pred[pred<=0] = -1
    
    right = np.sum(pred == y)
    wrong = len(pred)-right
            
    print("Good prediction: %i/%i (%f%%)\nWrong prediction: %i/%i (%f%%)"%
          (right, len(y), 100*right/len(y), wrong,  len(y), 100*wrong/len(y)))   

def perc_wrong_pred(y, tX, w_star):
    pred = np.dot(tX, w_star)

    pred[pred>0] = 1
    pred[pred<=0] = -1
    
    right = np.sum(pred == y)
    wrong = len(pred)-right
    
    return wrong/len(pred)
    
def perc_wrong_pred_logit(y, tX, w_star):
    pred = sigmoid(np.dot(tX, w_star))

    pred[pred>0.5] = 1
    pred[pred<=0.5] = 0
    
    right = np.sum(pred == y)
    wrong = len(pred)-right
    
    return wrong/len(pred)
            
def prediction_log(y, tX, w_star):

    pred = sigmoid(np.dot(tX, w_star))
    print(pred)

    pred[pred>0.5] = 1
    pred[pred<=0.5] = 0
    
    right = np.sum(pred == y)
    wrong = len(pred)-right   
            
    print("Good prediction: %i/%i (%f%%)\nWrong prediction: %i/%i (%f%%)"%
          (right, len(y), 100*right/len(y), wrong,  len(y), 100*wrong/len(y)))         

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.exp(-t))
  
def build_poly(x, degree):
    n_x = len(x)
    nbr_param = len(x[0])
    mat = np.zeros((n_x, (degree)*nbr_param))
        
    for j in range(nbr_param):
        for k in range(1,degree):
            mat[:, j*(degree)+k] = x[:,j]**k
			
    mat = np.hstack((np.ones((x.shape[0],1)), mat))
            
    return mat

def build_poly_multi_degree(x, degrees):
    N,nbr_param=np.shape(x)
    if len(degrees) != nbr_param:
        print("ERROR: the size of vector degrees must be equeal of numbere of parameters")
        return
    mat = np.ones(N)
    for i,degree in enumerate(degrees):
        if degree == 1:
            mat = np.c_[mat, x[:,i]]
        elif degree > 1:
            for d in range(int(degree)+1):
                mat=np.c_[mat, x[:,i]**d]
    return mat[:,1:]

def build_poly_cross_terms(x, degree, ct=False, sqrt=False, square=False):
    n_x = len(x)  
    nbr_param = len(x[0]) 
    
    nbr_ct = 0
    
    if ct:
        nbr_ct += int(nbr_param*(nbr_param-1)/2)
    
    if sqrt:
        nbr_ct += int(nbr_param*(nbr_param-1)/2)

    if square:
        nbr_ct += int(nbr_param*(nbr_param-1)/2)
        
    if degree > 1:
        mat = np.zeros((n_x, (degree+1)*nbr_param + nbr_ct))
            
        for j in range(nbr_param):
            for k in range(degree+1):
                mat[:, j*(degree+1)+k] = x[:,j]**k
                
                
        idx = (degree+1)*nbr_param
                
    elif degree==1:
        mat = np.zeros((n_x, nbr_param + nbr_ct))
        
        mat[:, :nbr_param] = x
                
        idx = nbr_param
    
                
    if ct:
        for l in range(nbr_param):
            for m in range(l+1, nbr_param):
                mat[:, idx] = x[:,l]*x[:,m]
                idx += 1
    
    if sqrt:
        for n in range(nbr_param):
            for o in range(n+1, nbr_param):
                mat[:, idx] = np.sqrt(np.abs(x[:,n]*x[:,o]))
                idx += 1 
                
    if square:
        for p in range(nbr_param):
            for q in range(p+1, nbr_param):
                mat[:, idx] = (x[:,p]*x[:,q])**2
                idx += 1
     
    return mat   

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
    Standardize the original data set. Using feature scaling:
                X = (X - Xmin) / (Xmax - Xmin)
                
    The predictions of -1 are set to 0.
                
    INPUT:
        x   - Samples matrix
        y   - Prediction vector
        
    OUTPUT:
        Rescaled version of the Samples matrix and the Prediction vector.
    """
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)    
    x = x - mean_x
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    # Add the column of ones at the beginnin
    tx = np.hstack((np.ones((x.shape[0],1)), x))
	# Replace all the -1 by 0
    y[y==-1] = 0
    return tx, y        