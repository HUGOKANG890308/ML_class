# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:50:44 2023

@author: EricMiao
"""
import structure as s

# some config
random_state = 0
test_size = 0.2
k_fold_num = 5
fs_model = s.SVC(kernel='rbf', C=10 )
fs_feature_num = 30
n_trials = 10

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

# read X, Y
df = s.pd.read_csv('data.csv')
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']

def prepare_X_Y():
    # train, validation, and test split
    x_train, x_test, y_train, y_test = s.train_test_split(X, Y, test_size = test_size, random_state = random_state)
    x_train, x_val, y_train, y_val = s.splitting_train_validation_StratifiedKFold(x_train, 
                                                                                  y_train, 
                                                                                  k_fold_num, 
                                                                                  random_state, 
                                                                                  our_shuffle = True)
    # standardize
    x_train, x_val, x_test = s.standardize(x_train, x_val, x_test, 'min_max')
    x_train = s.pd.DataFrame(x_train, columns = X.columns)
    x_val = s.pd.DataFrame(x_val, columns = X.columns)
    x_test = s.pd.DataFrame(x_test, columns = X.columns)
    return x_train, x_val, x_test, y_train, y_val, y_test

# feature selection
final_df = s.pd.DataFrame()
for fs_method_name, fs_method in s.tqdm(feature_selection_mtehod.items()):
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_X_Y()
    
    if fs_method_name == 'VIF + SFS':
        x_train, y_train = s.feature_selection(x_train, y_train, 'variance_inflation_factor', random_state)
    
    x_train, y_train = s.feature_selection(x_train, y_train, fs_method, fs_model, fs_feature_num, random_state)
    x_test = x_test[x_train.columns] ; x_val = x_val[x_train.columns]
    
    # training data Imbalance process
    imb_df = s.pd.DataFrame()
    for imb_method_name, method_no in imbalance_method.items():
        x_train, y_train = s.imblance_data(x_train, y_train, method_no, random_state)
    
        # Basic-ML
        models={'xgb_tuned': s.XGBClassifier(**s.study(method='xgb', n_trials = n_trials, X_train=x_train, y_train=y_train, X_val=x_val, y_val=y_val)), 
                'xgb':s.XGBClassifier(random_state = random_state),
                'rf_tuned': s.RandomForestClassifier(**s.study(method='rf', n_trials = n_trials, X_train=x_train, y_train=y_train, X_val=x_val, y_val=y_val)),
                'rf':s.RandomForestClassifier(random_state = random_state),
                'svm_tuned': s.SVC(**s.study(method='svm', n_trials = n_trials, X_train=x_train, y_train=y_train, X_val=x_val, y_val=y_val)),
                'svm':s.SVC(random_state = random_state)
                }
        
        ml_df = s.basic_ml(models, x_train, y_train, x_test, y_test )
        ml_df.insert(0, 'imbalance_method', imb_method_name)
        imb_df = s.pd.concat([imb_df, ml_df])
    
    imb_df = imb_df.reset_index(drop=True)
    
    
    imb_df.insert(0, 'feature_selection_method', fs_method_name)
    final_df = s.pd.concat([final_df, imb_df])

final_df = final_df.reset_index(drop=True)
final_df.to_csv('final_result.csv')
