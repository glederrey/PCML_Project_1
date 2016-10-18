# -*- coding: utf-8 -*-
"""a function used to compute the loss."""
import csv
import numpy as np

def load_data(data_path):
    """Loads data and returns y (labels), tX (features), ids (event ids) and the header"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]
    
    f = open(data_path, 'r')
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    
    return y, input_data, ids, headers
    
def perc_nan(tX):
    """This functions create an array with the percentage of NaN values in the column of tX"""
    nan = np.zeros(len(tX[0]))

    for i in range(len(tX[0])):
        for j in range(len(tX)):
            if tX[j,i] == -999:
                nan[i] = nan[i] + 1
                
    return nan/len(tX)
    
def delete_column_nan(nan, tX, headers, threshold = 0.65):
    """ This function will delete all the columns of tX with a percentage of NaN values
        higher than the threshold """
    
    nbr_col_deleted = 0
    index = []
    index_headers = []
    
    for i in range(len(nan)):
        if nan[i] > threshold:
            index.append(i)
            index_headers.append(i+2)
    
    tX = np.delete(tX, index, 1)
    nan = np.delete(nan, index)
    headers = np.delete(headers, index_headers)
            
    return tX, nan, headers
    
def replace_by_mean(tX, nan):
    """ We replace all the NaN values by the mean of the other values """
    for i in range(len(nan)):
        if nan[i] > 0:
            mean = 0
            nbr_val = 0
            for j in range(len(tX)):
                if tX[j,i] != -999:
                    mean = mean + tX[j,i]
                    nbr_val = nbr_val + 1
                    
            mean = mean/nbr_val
            for j in range(len(tX)):
                if tX[j,i] == -999:
                    tX[j,i] = mean
                    
    return tX
    
def write_data(output, y, tX, ids, headers):
    """Write the data into a CSV file"""
    with open(output, 'w') as csvfile:       
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=headers)
        writer.writeheader()
        for r1, r2, r3 in zip(ids, y, tX):
            dic = {'Id':int(r1),'Prediction':r2}
            for i in range(len(r3)):
                dic[headers[i+2]] = float(r3[i])
            writer.writerow(dic)
            
            
def clean_data(data_path, output):
    """ Clean the data using all the functions in this file """
    y, tX, ids, headers = load_data(data_path)
    nan = perc_nan(tX)
    tX, nan, headers = delete_column_nan(nan, tX, headers, 0.65)
    tX = replace_by_mean(tX, nan)
    write_data(output, y, tX, ids, headers)
    
    
    
        
                          
            
    
            
