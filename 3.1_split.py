import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold, LeaveOneOut
from sklearn.model_selection import StratifiedShuffleSplit

def splitting_train_test(csv_file_name, the_test_size=0.2, the_random_state=0):
    '''
    input:
        csv_file_name = 資料檔案名稱（包含.csv）; type = pandas dataframe
        the_test_size = size of training data; type = float; default = 0.2
        the_random_state = random state; type = int; default = 0 

    output:
        x_train = training data of x; type = pandas dataframe
        x_test = testing data of x; type = pandas dataframe
        y_train = training data of y; type = pandas dataframe
        y_test = testing data of y; type = pandas dataframe
    '''
    
    df = pd.read_csv(csv_file_name)
    X = df.drop(['Bankrupt?'], axis=1)
    Y = df['Bankrupt?']
    #print(f'X is {X},\n\n X.shape is {X.shape}')
    #print(f'Y is {Y},\n\n Y.shape is {Y.shape}')
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=the_test_size, random_state=the_random_state)
    #print(f'x_train is {x_train},\n\n x_train.shape is {x_train.shape}')
    #print(f'x_test is {x_test},\n\n x_test.shape is {x_test.shape}')
    #print(f'y_train is {y_train},\n\n y_train.shape is {y_train.shape}')
    #print(f'y_test is {y_test},\n\n y_test.shape is {y_test.shape}')
    
    return x_train, x_test, y_train, y_test

#x_train, x_test, y_train, y_test = splitting_train_test('data.csv')

def splitting_train_validation_HoldOut(csv_file_name, the_validation_size=0.2, the_random_state=0):
    '''
    input:
        csv_file_name = 資料檔案名稱（包含.csv）; type = pandas dataframe
        the_validation_size = data size of validaiton; type = float; default = 0.2
        the_random_state = random state; type = int; default = 0
    
    output:
        x_train = training data of x; type = pandas dataframe
        x_validation = validation data of x; type = pandas dataframe
        y_train = training data of y; type = pandas dataframe
        y_validation = validation data of y; type = pandas dataframe
    '''
    
    df = pd.read_csv(csv_file_name)
    X = df.drop(['Bankrupt?'], axis=1)
    Y = df['Bankrupt?']
    #print(f'X is {X},\n\n X.shape is {X.shape}')
    #print(f'Y is {Y},\n\n Y.shape is {Y.shape}')
    
    x_train, x_validation, y_train, y_validation = train_test_split(X, Y, test_size=the_validation_size, random_state=the_random_state)
    #print(f'x_train is {x_train},\n\n x_train.shape is {x_train.shape}')
    #print(f'x_validation is {x_validation},\n\n x_validation.shape is {x_validation.shape}')
    #print(f'y_train is {y_train},\n\n y_train.shape is {y_train.shape}')
    #print(f'y_validaiton is {y_validation},\n\n y_validation.shape is {y_validation.shape}')
    
    return x_train, x_validation, y_train, y_validation

#x_train, x_validation, y_train, y_validation = splitting_train_validation_HoldOut('data.csv')

def splitting_train_validation_KFold(csv_file_name, n, the_random_state=None, the_shuffle=False):
    '''
    input:
        csv_file_name = 資料檔案名稱（包含.csv）; type = pandas dataframe
        n = size of n_split; type = int
        the_random_state = random state; type = int; default = None
        the_shuffle = shuffle; type = bool; default = False
    
    output:
    '''
    df = pd.read_csv(csv_file_name)
    X = df.drop(['Bankrupt?'], axis=1)
    Y = df['Bankrupt?']
    #print(f'X is {X},\n\n X.shape is {X.shape}')
    #print(f'Y is {Y},\n\n Y.shape is {Y.shape}')

    KFold_result = KFold(n_splits=n, random_state=the_random_state, shuffle=the_shuffle)
    
    '''
    for train, validation in KFold_result.split(X, Y):
        print(' KFold Training tuple#:%s Valid tuple#:%s' % 
              (train, validation))
    '''        
    

    for train_index, validate_index in KFold_result.split(X, Y):
        #print("Train:", train_index, "Test:", test_index)
        x_train_raw, x_validation_raw = X.iloc[train_index], X.iloc[validate_index]
        y_train_raw, y_validation_raw = Y.iloc[train_index], Y.iloc[validate_index]
    #print(f'x_train_raw is {x_train_raw},\n\n x_train_raw.shape is {x_train_raw.shape}')
    #print(f'x_validation_raw is {x_validation_raw},\n\n x_validation_raw.shape is {x_validation_raw.shape}')
    #print(f'y_train_raw is {y_train_raw},\n\n y_train_raw.shape is {y_train_raw.shape}')
    #print(f'y_validaiton_raw is {y_validation_raw},\n\n y_validation_raw.shape is {y_validation_raw.shape}')
    
    x_train = x_train_raw.values
    x_validation = x_validation_raw.values
    y_train = y_train_raw.values
    y_validation = y_validation_raw.values
    
    return x_train, x_validation, y_train, y_validation

#x_train, x_validation, y_train, y_validation = splitting_train_validation_KFold('data.csv', 4, None)

def splitting_train_validation_StratifiedKFold(csv_file_name, n, the_random_state=None, the_shuffle=False):
    '''
    input:
        csv_file_name = 資料檔案名稱（包含.csv）; type = pandas dataframe
        n = size of n_split; type = int
        the_random_state = random state; type = int; default = None
        the_shuffle = shuffle; type = bool; default = False
    
    output:
        x_train = training data of x; type = pandas dataframe
        x_validation = validation data of x; type = pandas dataframe
        y_train = training data of y; type = pandas dataframe
        y_validation = validation data of y; type = pandas dataframe
    '''
    df = pd.read_csv(csv_file_name)
    X = df.drop(['Bankrupt?'], axis=1)
    Y = df['Bankrupt?']
    #print(f'X is {X},\n\n X.shape is {X.shape}')
    #print(f'Y is {Y},\n\n Y.shape is {Y.shape}')

    StratifiedKFold_result = StratifiedKFold(n_splits=n, random_state=the_random_state, shuffle=the_shuffle)

    '''
    for train, validation in StratifiedKFold_result.split(X, Y):
        print('SKFold Training tuple#:%s Valid tuple#:%s' % (train, validation))
    '''
    
    

    for train_index, validate_index in StratifiedKFold_result.split(X, Y):
        #print("Train:", train_index, "Test:", test_index)
        x_train_raw, x_validation_raw = X.iloc[train_index], X.iloc[validate_index]
        y_train_raw, y_validation_raw = Y.iloc[train_index], Y.iloc[validate_index]
    #print(f'x_train_raw is {x_train_raw},\n\n x_train_raw.shape is {x_train_raw.shape}')
    #print(f'x_validation_raw is {x_validation_raw},\n\n x_validation_raw.shape is {x_validation_raw.shape}')
    #print(f'y_train_raw is {y_train_raw},\n\n y_train_raw.shape is {y_train_raw.shape}')
    #print(f'y_validaiton_raw is {y_validation_raw},\n\n y_validation_raw.shape is {y_validation_raw.shape}')
    
    x_train = x_train_raw.values
    x_validation = x_validation_raw.values
    y_train = y_train_raw.values
    y_validation = y_validation_raw.values
    
    return x_train, x_validation, y_train, y_validation

#x_train, x_validation, y_train, y_validation = splitting_train_validation_StratifiedKFold('data.csv', 4, None)

def splitting_train_validation_LeaveOneOut(csv_file_name, n, the_random_state=None, the_shuffle=False):
    '''
    input:
        csv_file_name = 資料檔案名稱（包含.csv）; type = pandas dataframe
        n = size of n_split; type = int
        the_random_state = random state; type = int; default = None
        the_shuffle = shuffle; type = bool; default = False
    
    output:
        x_train = training data of x; type = pandas dataframe
        x_validation = validation data of x; type = pandas dataframe
        y_train = training data of y; type = pandas dataframe
        y_validation = validation data of y; type = pandas dataframe
    '''
    df = pd.read_csv(csv_file_name)
    X = df.drop(['Bankrupt?'], axis=1)
    Y = df['Bankrupt?']
    #print(f'X is {X},\n\n X.shape is {X.shape}')
    #print(f'Y is {Y},\n\n Y.shape is {Y.shape}')
    
    LeaveOneOut_result = LeaveOneOut()

    
    for train, validation in LeaveOneOut_result.split(X, Y):
        print('LOO Training tuple#:%s Valid tuple#:%s' % (train, validation))
    


    #return x_train, x_validation, y_train, y_validation

#x_train, x_validation, y_train, y_validation = splitting_train_validation_StratifiedKFold('data.csv', 4, None)


