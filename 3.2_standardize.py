import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def standardize(training_data, validation_data, testing_data, method):
    '''
    input:
        training_data = input training data(data after spiltting); type = pandas dataframe
        validation_data = input validation data(data after spiltting); type = pandas dataframe
        testing_data = input testing data(data after spiltting); type = pandas dataframe
        method = input method you use to standardize data; type = string
                1. standardize the data
                2. minmax the data

    output:
        training_data = training data after standardize; type = pandas dataframe
        validation_data = validation data after standardize; type = pandas dataframe
        testing_data = testing data after standardize; type = pandas dataframe
    '''

    if method == 'z-score normalization':
        StandardScaler().fit(training_data)
        StandardScaler().fit(validation_data)
        StandardScaler().fit(testing_data)
        
        zscore_training = StandardScaler().transform(training_data)
        zscore_validation = StandardScaler().transform(validation_data)
        zscore_testing = StandardScaler().transform(testing_data)
        #print(zscore_training)
        #print(zscore_validation)
        #print(zscore_testing)
        
        training_data  = (zscore_training  + 3)/6
        validation_data  = (zscore_validation  + 3)/6
        testing_data  = (zscore_testing  + 3)/6
        #print(training_data)
        #print(validaiton_data)
        #print(testing_data)
        
    elif method == 'min max':
        MinMaxScaler().feature_range = (0, 1) #這句是否需要？
        
        MinMaxScaler().fit(training_data)
        MinMaxScaler().fit(validation_data)
        MinMaxScaler().fit(testing_data)
        
        training_data =  MinMaxScaler().transform(training_data)
        validation_data =  MinMaxScaler().transform(validation_data)
        testing_data =  MinMaxScaler().transform(testing_data)
        #print(training_data)
        #print(validaiton_data)
        #print(testing_data)
        
        
    
    '''
    # LASSO
    if method == 'standardize the data':
        training_data  = StandardScaler().fit_transform(training_data)
        validation_data  = StandardScaler().fit_transform(validation_data)
        testing_data  = StandardScaler().fit_transform(testing_data)
    else:
        # build the scaler model
        scaler = MinMaxScaler()
        # fit using the train set
        scaler.fit(X)
        # transform the test test
        X_scaled = scaler.transform(X)
        # Verify minimum value of all features
        X_scaled.min(axis=0)
        # array([0., 0., 0., 0.])
        # Verify maximum value of all features
        X_scaled.max(axis=0)
        # array([1., 1., 1., 1.])
        # Manually normalise without using scikit-learn
        X_manual_scaled = (X — X.min(axis=0)) / (X.max(axis=0) — X.min(axis=0))
        # Verify manually VS scikit-learn estimation
        print(np.allclose(X_scaled, X_manual_scaled))
        #True
    '''
    
    return training_data, validation_data, testing_data
