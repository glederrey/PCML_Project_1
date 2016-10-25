# -*- coding: utf-8 -*-
"""
Code for the PCA analysis
"""

import numpy as np

def mean_vector(tX):
    return np.mean(tX,axis=0)
    
def covariance_mat(tX):
    arrays = []
    for i in range(len(tX[0])):
        arrays.append(tX[i,:])
        
    return np.cov(arrays)
    
def pca(tX, nbr):

    # Nbr param
    nbr_param = len(tX[0])
    # Number of entries
    nbr_entries = len(tX)
    # Mean vector
    mean_v = mean_vector(tX)
    # Covariance matrix
    cov_mat = covariance_mat(tX)
    # eigen values and eigen vectors
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    
    # Check there's no problem with the eigenvalues/eigenvectors
    for i in range(len(eig_val)):
        eigv = eig_vec[:,i].reshape(1,nbr_param).T
        np.testing.assert_array_almost_equal(cov_mat.dot(eigv), eig_val[i] * eigv,
                                             decimal=6, err_msg='', verbose=True)    

    for ev in eig_vec:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=1) 

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    vect = []
    for i in range(nbr):
        vect.append(eig_pairs[i][1].reshape(nbr_param,1))
    
    matrix_w = np.hstack(vect)
    matrix_w = np.real(matrix_w)
    
    transformed = matrix_w.T.dot(tX.T)
    
    return transformed.T
    