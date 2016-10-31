# -*- coding: utf-8 -*-
"""
    Function to use to get the best results on Kaggle
"""

import argparse
from helpers_run import *

def main(da, cv):

    # Name of the train file
    TRAIN = 'train.csv'
    try:
        open(TRAIN, 'r')
    except:
        raise NameError('Cannot open file %s! Are you sure it exists in this directory' % TRAIN)

    # Name of the test file
    TEST = 'test.csv'
    try:
        open(TEST, 'r')
    except:
        raise NameError('Cannot open file %s! Are you sure it exists in this directory' % TEST)

    # Name of the training data
    TRAINING_DATA = ['train_jet_0_wout_mass.csv', 'train_jet_0_with_mass.csv',
                     'train_jet_1_wout_mass.csv', 'train_jet_1_with_mass.csv',
                     'train_jet_2_wout_mass.csv', 'train_jet_2_with_mass.csv',
                     'train_jet_3_wout_mass.csv', 'train_jet_3_with_mass.csv']

    # Name of the test data                   
    TESTING_DATA = ['test_jet_0_wout_mass.csv', 'test_jet_0_with_mass.csv',
                    'test_jet_1_wout_mass.csv', 'test_jet_1_with_mass.csv',
                    'test_jet_2_wout_mass.csv', 'test_jet_2_with_mass.csv',
                    'test_jet_3_wout_mass.csv', 'test_jet_3_with_mass.csv']
    if da:
        data_analysis_splitting(TRAIN, TEST, TRAINING_DATA, TESTING_DATA)

    if cv:
        perc_right_pred, degree_star, lambda_star = cross_validation(TRAINING_DATA, verbose=True)
        print(u'Percentage of right pred on training set: {0:f}'.format(perc_right_pred))
        print('degree_star = ', degree_star)
        print('lambda_star = ', lambda_star)
    else:
        # Hardcoded values
        degree_star = [12, 9, 7, 9, 10, 10, 8, 9]
        lambda_star = [9e-06, 0.0212, 1.65e-05, 0.00027, 2.42e-06, 0.000309, 4e-05, 3.63e-10]

    # We define the hardcoded booleans for the cross-terms
    ct = [False, True, False, True, True, True, False, True]
    sqrt = [True, True, True, True, False, True, False, True]
    square = [False, True, False, True, False, True, True, False]

    """ TRAINING """
    weights, prediction_train = training(TRAINING_DATA, degree_star, lambda_star, ct, sqrt, square)

    print(u'\nIn total, there was {0:2f}% of good predictions on the training set.\n'.format(prediction_train))

    """ TESTING """
    test(TESTING_DATA, degree_star, ct, sqrt, square, weights, 'RR_8models_10foldCV_CT.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' This function implements the best run of team 81 on Kaggle. ')
    parser.add_argument('-da', action='store_true', help='Performs the Data Analysis and Splitting', default=False)
    parser.add_argument('-cv', action='store_true',
                        help='Performs the Cross-Validation. If not called, it will use hardcoded values.',
                        default=False)
    args = parser.parse_args()

    main(args.da, args.cv)
