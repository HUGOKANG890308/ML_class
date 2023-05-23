# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:57:04 2023

@author: AllenWu
"""

import structure as s

# Difine parameter
random_state = 0
shuffle = False
input_size = None
n_epochs = 10
batch_size = 128
n_trials = 5

feature_selection_mtehod = {'VIF': 'variance_inflation_factor', 
                            'VIF + SFS': 'Sequential Feature Selection', 
                            'Whole_Feature': 'raw'
                            }

imbalance_method = {'ROS': 1, 
                    'RUS': 2, 
                    'SMOTE': 3, 
                    'ADASYN': 4, 
                    'SMOTEENN': 5, 
                    'Without_balance': 6
                    }

fs_model = s.SVC(kernel='rbf', C=10 )
fs_method_name = 'VIF + SFS'
fs_method = feature_selection_mtehod[fs_method_name]
fs_feature_num = 30

imb_method_name = 'ROS'
imbalance_num = imbalance_method[imb_method_name]


df = s.pd.read_csv('data.csv')
X = df.drop(['Bankrupt?'], axis=1)
y = df['Bankrupt?']


# Splite data
X_train, X_test, y_train, y_test = s.train_test_split(X, y, test_size=0.2, random_state=random_state)
X_train, X_valid, y_train, y_valid = s.splitting_train_validation_StratifiedKFold(X_train, y_train, 5)

# Standardize
X_train, X_valid, X_test = s.standardize(X_train, X_valid, X_test, 'min_max')

# Conver data to df
X_train = s.pd.DataFrame(X_train, columns = X.columns)
X_valid = s.pd.DataFrame(X_valid, columns = X.columns)
X_test = s.pd.DataFrame(X_test, columns = X.columns)
 

# Feature selection
X_train, y_train = s.feature_selection(X_train, y_train, 'variance_inflation_factor', random_state)
X_train, y_train = s.feature_selection(X_train, y_train, fs_method, fs_model, fs_feature_num, random_state)
X_test = X_test[X_train.columns] ; X_valid = X_valid[X_train.columns]
input_size = len(X_train.columns) # Difine Input_size for nn model


# solve imbalance 
X_train, y_train = s.imblance_data(X_train, y_train, imbalance_num, random_state = random_state)

X_train = s.np.array(X_train)
X_valid = s.np.array(X_valid)
X_test = s.np.array(X_test)
y_train = s.np.array(y_train)
y_valid = s.np.array(y_valid)
y_test = s.np.array(y_test)

input_size = X_train.shape[1]

train_loader, valid_loader, test_loader = s.convert_to_DataLoader(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size)
nn_df = s.Training_nn(train_loader, valid_loader, test_loader, input_size, n_epochs, batch_size, n_trials)
nn_df.insert(0, 'imbalance_method', imb_method_name)
nn_df.insert(0, 'feature_selection_method', fs_method_name)

