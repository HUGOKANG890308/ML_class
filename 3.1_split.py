import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold, LeaveOneOut
from sklearn.model_selection import StratifiedShuffleSplit


df = pd.read_csv('data.csv')
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']
print(f'X is {X}, X.shape is {X.shape}\n\n')
print(f'Y is {Y}, Y.shape is {Y.shape}\n\n')

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = our_test_size, random_state = our_random_state)


#x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = our_validation_size, random_state = our_random_state)



def splitting_train_validation_StratifiedKFold(X, Y, n, our_random_state = None, our_shuffle = False):
    '''
    切出stratifiedkfold的train validation data
    input:
        X = 資料 (非 target) ; type = pandas dataframe
        Y = 資料 (target) ; type = pandas dataframe
        method = the method that used to split training data and validation data ; type = string
            1. KFold
            2. StratifiedKFold
            3. LeaveOneOut
        n = size of n_split ; type = int
        our_random_state = random state ; type = int ; default = None
        our_shuffle = shuffle ; type = bool ; default = False
    
    output:
        x_train = training data of x ; type = pandas dataframe
        x_validation = validation data of x ; type = pandas dataframe
        y_train = training data of y ; type = pandas dataframe
        y_validation = validation data of y ; type = pandas dataframe
    '''   
   #可以用在unbalanced data上 因為在分的時候會讓unbalanced data平均分在每個fold裡
   StratifiedKFold_result = StratifiedKFold(n_splits = n, random_state = our_random_state, shuffle = our_shuffle)
    
   for train_index, validate_index in StratifiedKFold_result.split(X, Y):
       print("Train index:", train_index, "Test index:", validate_index)
       x_train_raw, x_validation_raw = X.iloc[train_index], X.iloc[validate_index]
       y_train_raw, y_validation_raw = Y.iloc[train_index], Y.iloc[validate_index]
    
    print('/n/n')
    x_train = x_train_raw.values
    x_validation = x_validation_raw.values
    y_train = y_train_raw.values
    y_validation = y_validation_raw.values
    
    #print(f'x_train is {x_train}, x_train.shape is {x_train.shape}\n\n')
    #print(f'x_validation is {x_validation}, x_validation.shape is {x_validation.shape}\n\n')
    #print(f'y_train is {y_train}, y_train.shape is {y_train.shape}\n\n')
    #print(f'y_validation is {y_validation}, y_validation.shape is {y_validation.shape}\n\n')
        
    return x_train, x_validation, y_train, y_validation
