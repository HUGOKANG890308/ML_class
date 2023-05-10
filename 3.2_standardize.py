import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def standardize(x_training_data, x_validation_data, x_testing_data, method): 
    '''
    標準化資料，只丟x進來
    input:
        x_training_data = input training data(data after spiltting) ; type = pandas dataframe
        x_validation_data = input validation data(data after spiltting) ; type = pandas dataframe
        x_testing_data = input testing data(data after spiltting) ; type = pandas dataframe
        method = input method you use to standardize data ; type = string
                1. standardize the data
                2. minmax the data

    output:
        x_training_data = training data after standardize ; type = pandas dataframe
        x_validation_data = validation data after standardize ; type = pandas dataframe
        x_testing_data = testing data after standardize ; type = pandas dataframe
    '''

    if method == 'z_score_normalization':
        temp = StandardScaler()
        
        #training_data  = (zscore_training  + 3)/6
        #validation_data  = (zscore_validation  + 3)/6
        #testing_data  = (zscore_testing  + 3)/6
        
    elif method == 'min_max':
        temp =  MinMaxScaler()
    else:
        print('wrong input of method /n/n')
        exit      
  
    try:
        x_training_data  = temp.fit_transform(x_training_data)
        x_validation_data  = temp.transform(x_validation_data)
        x_testing_data  = temp.transform(x_testing_data)
    except:
        print('please use right method /n/n')
    
    return x_training_data, x_validation_data, x_testing_data
    
  
    
