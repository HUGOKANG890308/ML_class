import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def standardize(training_data, validation_data, testing_data, method):
    '''
    input:
        training_data = input training data(data after spiltting) ; type = pandas dataframe
        validation_data = input validation data(data after spiltting) ; type = pandas dataframe
        testing_data = input testing data(data after spiltting) ; type = pandas dataframe
        method = input method you use to standardize data ; type = string
                1. standardize the data
                2. minmax the data

    output:
        training_data = training data after standardize ; type = pandas dataframe
        validation_data = validation data after standardize ; type = pandas dataframe
        testing_data = testing data after standardize ; type = pandas dataframe
    '''

    if method == 'z-score normalization':
        
        zscore_training  = StandardScaler().fit_transform(training_data)
        zscore_validation  = StandardScaler().fit_transform(validation_data)
        zscore_testing  = StandardScaler().fit_transform(testing_data)
        print(zscore_training)
        print(zscore_validation)
        print(zscore_testing)
        
        training_data  = (zscore_training  + 3)/6
        validation_data  = (zscore_validation  + 3)/6
        testing_data  = (zscore_testing  + 3)/6
        
    elif method == 'min max':
        MinMaxScaler().feature_range = (0, 1) #這句是否需要？
        
        training_data  = StandardScaler().fit_transform(training_data)
        validation_data  = StandardScaler().fit_transform(validation_data)
        testing_data  = StandardScaler().fit_transform(testing_data)
    else:
        print('wrong input of method /n/n')
        exit        
        
    print(training_data)
    print(validation_data)
    print(testing_data)
    
    return training_data, validation_data, testing_data
    
    '''
    else:
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        X_scaled.min(axis=0)
        X_scaled.max(axis=0)
        X_manual_scaled = (X — X.min(axis=0)) / (X.max(axis=0) — X.min(axis=0))
        print(np.allclose(X_scaled, X_manual_scaled)) # True
    '''
    
