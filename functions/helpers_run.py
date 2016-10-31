# -*- coding: utf-8 -*-
"""
    Helpers for the function run.py
"""

from helpers import *
from implementations import ridge_regression
import numpy as np
import csv

""" ----------- MAIN FUNCTIONS ----------- """


def data_analysis_splitting(TRAIN, TEST, TRAINING_DATA, TESTING_DATA):
    """
        This long function is used for the data analysis and splitting.
        This function will split the data according as the description made in the report.

        We first split in 4 by the number of jets, then we split by the remaining NaNs in the
        first column. Then, we can write the new files.
    """

    print('START THE DATA ANALYSIS / SPLITTING FOR DATA-SETS')
    print('  Load the data. It may take a few seconds.')

    # First we load the data
    headers = get_headers(TRAIN)
    y_train, tx_train, ids_train = load_csv_data(TRAIN)
    y_test, tx_test, ids_test = load_csv_data(TEST)

    # Start the loop for the four kind of jets
    for jet in range(4):
        print("  Cleaning for Jet {0:d}".format(jet))

        # Get the new matrix with only the same jets
        # The information about the number of jets is in column 22
        tx_jet_train = tx_train[tx_train[:, 22] == jet]
        tx_jet_test = tx_test[tx_test[:, 22] == jet]

        # Cut the predictions for the same jet
        y_jet_train = y_train[tx_train[:, 22] == jet]
        y_jet_test = y_test[tx_test[:, 22] == jet]

        # Cut the ids for the same jet
        ids_jet_train = ids_train[tx_train[:, 22] == jet]
        ids_jet_test = ids_test[tx_test[:, 22] == jet]

        # Delete column 22 in Sample matrix
        tx_jet_train = np.delete(tx_jet_train, 22, 1)
        tx_jet_test = np.delete(tx_jet_test, 22, 1)

        # Delete column 24 (column 1 is ids, column 2 is pred) in headers
        headers_jet = np.delete(headers, 24)

        # Get all the columns with only NaNs
        nan_jet = np.ones(tx_jet_train.shape[1], dtype=bool)
        header_nan_jet = np.ones(tx_jet_train.shape[1] + 2, dtype=bool)
        for i in range(tx_jet_train.shape[1]):
            array = tx_jet_train[:, i]
            nbr_nan = len(array[array == -999])
            if nbr_nan == len(array):
                nan_jet[i] = False
                header_nan_jet[i + 2] = False

        # For Jet 0, there is a really big outlier in the column 3. So, we will remove it
        if jet == 0:
            to_remove = (tx_jet_train[:, 3] < 200)

        """ Start removing values """

        if jet == 0:
            tx_jet_train = tx_jet_train[to_remove, :]
            y_jet_train = y_jet_train[to_remove]
            ids_jet_train = ids_jet_train[to_remove]

            # We also remove the last column which is full of 0
            nan_jet[-1] = False
            header_nan_jet[-1] = False

        # Delete the columns in tX and headers
        tx_jet_train = tx_jet_train[:, nan_jet]
        tx_jet_test = tx_jet_test[:, nan_jet]

        headers_jet = headers_jet[header_nan_jet]

        # Get the NaNs in the mass
        nan_mass_jet_train = (tx_jet_train[:, 0] == -999)
        nan_mass_jet_test = (tx_jet_test[:, 0] == -999)
        header_nan_mass_jet = np.ones(len(headers_jet), dtype=bool)
        header_nan_mass_jet[2] = False

        # Write the files
        write_data(TRAINING_DATA[2 * jet], y_jet_train[nan_mass_jet_train], tx_jet_train[nan_mass_jet_train, :][:, 1:],
                   ids_jet_train[nan_mass_jet_train], headers_jet[header_nan_mass_jet], 'train')

        write_data(TRAINING_DATA[2 * jet + 1], y_jet_train[~nan_mass_jet_train], tx_jet_train[~nan_mass_jet_train, :],
                   ids_jet_train[~nan_mass_jet_train], headers_jet, 'train')

        write_data(TESTING_DATA[2 * jet], y_jet_test[nan_mass_jet_test], tx_jet_test[nan_mass_jet_test, :][:, 1:],
                   ids_jet_test[nan_mass_jet_test], headers_jet[header_nan_mass_jet], 'test')

        write_data(TESTING_DATA[2 * jet + 1], y_jet_test[~nan_mass_jet_test], tx_jet_test[~nan_mass_jet_test, :],
                   ids_jet_test[~nan_mass_jet_test], headers_jet, 'test')

    print("FINISHED SPLITTING THE DATA-SETS")


def cross_validation(TRAINING_DATA, verbose):
    lambda_star = []
    degree_star = []
    perc_right_pred = 0
    nbr_labels = 0

    degrees_poly = np.arange(8, 13)
    degrees_lambdas = np.arange(-10, 5)
    k_fold = 10
    digits = 3

    for idx, data in enumerate(TRAINING_DATA):
        # Print that we start the training
        print("Cross-validation with file %s" % data)
        print("-----------------------------------------------------")
        # Load the file
        y_train, x_train, ids_train = load_csv_data(data)

        # Do the Cross Validation for the ridge regression
        min_loss, deg, lamb = cv(y_train, x_train,
                                  degrees_lambdas, degrees_poly,
                                  k_fold, digits, verbose=verbose)
        # Print some interesting values
        print("  Max pred = %f" % (1 - min_loss))
        print("  Lambda* = %10.3e" % lamb)
        print("  Degree* = %i" % deg)
        print("\n")
        lambda_star.append(lamb)
        degree_star.append(deg)

        perc_right_pred += len(y_train) * (1 - min_loss)
        nbr_labels += len(y_train)

    perc_right_pred = perc_right_pred / nbr_labels
    return perc_right_pred, degree_star, lambda_star


def training(TRAINING_DATA, degree_star, lambda_star, ct, sqrt, square):
    """
        Train on the data with the degree_star and lambda_star found by the cross-validation.

        At the end, we return the best weights and the percentage of correct prediction on the
        training set.
    """
    weights = []
    total = 0
    mean = 0
    for idx, data in enumerate(TRAINING_DATA):
        # Print that we start the training
        print(u'Training with file {0:s}'.format(data))
        print(u'-----------------------------------------------------')
        # Load the file
        y_train, x_train, ids_train = load_csv_data(data)

        # Ridge Regression to get the best weights
        tx_train = build_poly_cross_terms(x_train, degree_star[idx],
                                          ct=ct[idx], sqrt=sqrt[idx], square=square[idx])

        w_star, _ = ridge_regression(y_train, tx_train, lambda_star[idx])
        # Get the percentage of wrong prediction
        val = perc_wrong_pred(y_train, tx_train, w_star)
        print(u'  Good prediction: {0:f}'.format(100. * (1. - val)))
        # Update the total number of entries tested/trained
        total += len(y_train)
        # Update the mean value of good prections
        mean += (1 - val) * len(y_train)

        weights.append(w_star)

    return weights, 100 * mean / total


def test(TESTING_DATA, degree_star, ct, sqrt, square, weights, output_name):
    """
        Use the degree_star and lambda_star from the Cross-Validation as well as the weights
        from the Training to test on the TEST data-set. It will write the file of predictions
        ready to be submitted on Kaggle.
    """

    y_pred = []
    ids_pred = []

    # Test on all the TEST data
    for idx, data in enumerate(TESTING_DATA):
        # Print that we start the testing
        print("Testing with file %s" % data)
        print("-----------------------------------------------------")
        # Recreate the file
        data_file = data
        # Load the file
        _, x_test, ids_test = load_csv_data(data_file)
        # Build the polynomial
        tx_test = build_poly_cross_terms(x_test, degree_star[idx],
                                         ct=ct[idx], sqrt=sqrt[idx], square=square[idx])

        # Predict the labels
        y_pred.append(predict_labels(weights[idx], tx_test))
        ids_pred.append(ids_test)

    # Put all the prediction together given the IDs
    ids = []
    pred = []

    idx = min(ids_pred[:][0])

    length = np.sum(len(i) for i in y_pred)

    print("Concatenate the predictions.")

    for i in range(length):
        for j in range(len(TESTING_DATA)):
            if len(ids_pred[j]) > 0:
                if ids_pred[j][0] == idx:
                    ids.append(idx)
                    pred.append(y_pred[j][0])
                    ids_pred[j] = np.delete(ids_pred[j], 0)
                    y_pred[j] = np.delete(y_pred[j], 0)
                    break

        if i % 100000 == 0:
            print(u'  {0:d}/{1:d} concatenated'.format(i, length))

        idx += 1

    # Transform the variables in ndarray
    pred = np.array(pred)
    ids = np.array(ids)

    # Write the file of predictions
    create_csv_submission(ids, pred, output_name)

    print(u'Data are ready to be submitted!')


""" -----------HELPERS FUNCTIONS ----------- """


def get_headers(data_path):
    """
        Get the headers from the file given in parameter
    """

    f = open(data_path, 'r')
    reader = csv.DictReader(f)
    headers = reader.fieldnames

    return headers


def write_data(output, y, tx, ids, headers, type_):
    """
        Write the data into a CSV file
    """
    with open(output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=headers)
        writer.writeheader()
        if type_ == 'train':
            for r1, r2, r3 in zip(ids, y, tx):
                if r2 == 1:
                    pred = 's'
                elif r2 == -1:
                    pred = 'b'
                else:
                    pred = r2
                dic = {'Id': int(r1), 'Prediction': pred}
                for i in range(len(r3)):
                    dic[headers[i + 2]] = float(r3[i])
                writer.writerow(dic)
        elif type_ == 'test':
            for r1, r3 in zip(ids, tx):
                dic = {'Id': int(r1), 'Prediction': '?'}
                for i in range(len(r3)):
                    dic[headers[i + 2]] = float(r3[i])
                writer.writerow(dic)


def build_poly_cross_terms(x, degree, ct=False, sqrt=False, square=False):
    """
        Build the polynomial basis of the sample matrix x with degree d.
        If the boolean are set to True, we add the corresponding cross-terms
    """

    n_x = len(x)
    nbr_param = len(x[0])

    # Number of parameters for the cross-terms
    nbr_ct = 0

    if ct:
        nbr_ct += int(nbr_param * (nbr_param - 1) / 2)
    if sqrt:
        nbr_ct += int(nbr_param * (nbr_param - 1) / 2)
    if square:
        nbr_ct += int(nbr_param * (nbr_param - 1) / 2)

    # If the degree is 1, we will not add the column of ones due to degree 0.
    # Otherwise, we use the definition seen in class.
    if degree > 1:
        mat = np.zeros((n_x, (degree + 1) * nbr_param + nbr_ct))

        for j in range(nbr_param):
            for k in range(degree + 1):
                mat[:, j * (degree + 1) + k] = x[:, j] ** k

        # Prepare for the cross-terms
        idx = (degree + 1) * nbr_param

    elif degree == 1:
        mat = np.zeros((n_x, nbr_param + nbr_ct))

        mat[:, :nbr_param] = x

        # Prepare for the cross-terms
        idx = nbr_param

    # Add the cross-terms by multiplication of pairs
    if ct:
        for l in range(nbr_param):
            for m in range(l + 1, nbr_param):
                mat[:, idx] = x[:, l] * x[:, m]
                idx += 1

    # Add the cross-terms by sqrt of the multiplication of pairs
    if sqrt:
        for n in range(nbr_param):
            for o in range(n + 1, nbr_param):
                mat[:, idx] = np.sqrt(np.abs(x[:, n] * x[:, o]))
                idx += 1

    # Add the cross-terms by the square of the multiplication of pairs
    if square:
        for p in range(nbr_param):
            for q in range(p + 1, nbr_param):
                mat[:, idx] = (x[:, p] * x[:, q]) ** 2
                idx += 1

    return mat


def perc_wrong_pred(y, tx, w_star):
    """
        Return the percentage of wrong predictions (between 0 and 1)
    """

    pred = np.dot(tx, w_star)

    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    right = np.sum(pred == y)
    wrong = len(pred) - right

    return float(wrong) / float(len(pred))


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
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})

def cv(y, tx, deg_lambdas, degrees, k_fold, digits, verbose=True, seed=1):
    """
        K-fold cross validation for the Ridge Regression
    """

    assert digits > 0, 'digits must be at least 1'
    if verbose:
        print("  Start the %i-fold Cross Validation!" % k_fold)

    # Prepare the matrix of loss
    loss_deg = np.zeros(len(degrees))
    lambs_star = np.zeros(len(degrees))

    # Split data in k-fold
    k_indices = build_k_indices(y, k_fold, seed)

    # Loop on the degrees
    for ideg, deg in enumerate(degrees):
        if verbose:
            print("  Start degree %i" % (deg))
        deg = int(deg)

        # Create the matrices
        mats_train, pred_train, mats_test, pred_test = create_matrices(y, tx, k_indices, deg)

        # First, we find the best lambdas in the first digit
        size = 9 * len(deg_lambdas)
        loss_lmbd = np.zeros(size)
        lmbd = np.zeros(size)

        idx = 0
        # Loop on the degrees of lambdas
        for idlamb, dlamb in enumerate(deg_lambdas):

            # loop on the first digit
            for i in range(1, 10):
                lambda_ = i * (10 ** int(dlamb))
                lmbd[idx] = lambda_

                loss_te = []
                # Loop on the k indices
                for k in range(k_fold):
                    try:
                        w_star, _ = ridge_regression(pred_train[k], mats_train[k], lambda_)
                        loss_te.append(perc_wrong_pred(pred_test[k], mats_test[k], w_star))
                    except:
                        loss_te.append(np.inf)

                loss_lmbd[idx] = np.median(loss_te)
                idx += 1

        # Now, we go deeper for the digits of lambda
        for dg in range(2, digits + 1):

            # find the lambda corresponding to the minimum value of loss
            idx_min = np.argmin(loss_lmbd)

            # Depending on the position of the lambda, we choose an interval inside which
            # we will go deeper
            if idx_min == 0:
                loss_lmbd = np.zeros(11)
                lmbd = np.linspace(lmbd[0], lmbd[1], 11)

            elif idx_min == len(loss_lmbd) - 1:
                loss_lmbd = np.zeros(11)
                lmbd = np.linspace(lmbd[-2], lmbd[-1], 11)
            else:
                loss_lmbd = np.zeros(21)
                lmbd = np.linspace(lmbd[idx_min - 1], lmbd[idx_min + 1], 21)

            # Test the new lambda
            for ilmbd in range(len(lmbd)):

                loss_te = []
                # Loop on the k indices
                for k in range(k_fold):
                    try:
                        w_star, _ = ridge_regression(pred_train[k], mats_train[k], lmbd[ilmbd])
                        loss_te.append(perc_wrong_pred(pred_test[k], mats_test[k], w_star))
                    except:
                        loss_te.append(np.inf)

                loss_lmbd[ilmbd] = np.median(loss_te)

        # Get the best lambda for the actual degree
        idx_min = np.argmin(loss_lmbd)
        if verbose:
            print("  Finished Degree %i. Best lambda is %10.3e with percentage wrong pred %f" % (
                deg, lmbd[idx_min], loss_lmbd[idx_min]))
        loss_deg[ideg] = loss_lmbd[idx_min]
        lambs_star[ideg] = lmbd[idx_min]

        if verbose:
            print("  --------------------")

    if verbose:
        print("%  i-fold Cross Validation finished!\n" % k_fold)

    # Return the best lambda, the best degree and the min error
    idx_min = np.argmin(loss_deg)
    lambda_star = lambs_star[idx_min]
    degree_star = degrees[idx_min]
    min_loss = loss_deg[idx_min]

    return min_loss, degree_star, lambda_star


def create_matrices(y, tx, k_indices, degree):
    """"
        Create all the matrices for the k_indices.
        It makes you win some time.
    """
    # Prepare the lists
    mats_test = []
    pred_test = []
    mats_train = []
    pred_train = []
    # Loop on the k_indices
    for k in range(len(k_indices)):
        # Values for the test
        tx_test = tx[k_indices[k]]
        pred_test.append(y[k_indices[k]])

        # Get all the indices that are not in the test data
        train_indices = []
        for i in range(len(k_indices)):
            if i != k:
                train_indices.append(k_indices[i])

        train_indices = np.array(train_indices)
        train_indices = train_indices.flatten()

        # Values for the train
        tx_train = tx[train_indices]
        pred_train.append(y[train_indices])

        # Build the polynomials functions
        tx_train = build_poly_cross_terms(tx_train, degree)
        tx_test = build_poly_cross_terms(tx_test, degree)

        mats_test.append(tx_test)
        mats_train.append(tx_train)

    return mats_train, pred_train, mats_test, pred_test
