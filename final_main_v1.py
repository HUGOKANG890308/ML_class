# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:50:44 2023

@author: EricMiao
"""
import datetime
import structure as s

today = datetime.date.today()

# some config
random_state = 0
test_size = 0.2
k_fold_num = 3
fs_model = s.SVC(kernel='rbf', C=10 )
fs_feature_num = 30
n_trials = 1

feature_selection_mtehod = {'VIF': 'variance_inflation_factor', 
                            # 'VIF + SFS': 'Sequential Feature Selection', 
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

# train, test split
X_train, x_test, Y_train, y_test = s.train_test_split(X, Y, test_size = test_size, random_state = random_state)

# k_fold loop
cv_score = list()
sskf = s.StratifiedKFold(n_splits=k_fold_num, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(sskf.split(X_train, Y_train)):
    print("Train index:", train_index, "Test index:", val_index)
    x_train, x_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train, y_val = Y_train.iloc[train_index], Y_train.iloc[val_index]
    
    # standardize
    x_train, x_val, x_test = s.standardize(x_train, x_val, x_test, 'min_max')
    x_train = s.pd.DataFrame(x_train, columns = X.columns)
    x_val = s.pd.DataFrame(x_val, columns = X.columns)
    x_test = s.pd.DataFrame(x_test, columns = X.columns)

    # feature selection
    fs_df = s.pd.DataFrame()
    for fs_method_name, fs_method in feature_selection_mtehod.items():
        if fs_method_name == 'VIF + SFS':
            x_train_select, y_train_select = s.feature_selection(x_train, y_train, 'variance_inflation_factor', random_state)
        
        x_train_select, y_train_select = s.feature_selection(x_train, y_train, fs_method, fs_model, fs_feature_num, random_state)
        x_test_select = x_test[x_train_select.columns] ; x_val_select = x_val[x_train_select.columns]
        
        # training data Imbalance process
        imb_df = s.pd.DataFrame()
        for imb_method_name, method_no in imbalance_method.items():
            x_train_select, y_train_select = s.imblance_data(x_train_select, y_train_select, method_no, random_state)
        
            # Basic-ML
            models={'xgb_tuned': s.XGBClassifier(**s.study(method='xgb', n_trials = n_trials, X_train=x_train_select, y_train=y_train_select, X_val=x_val_select, y_val=y_val)), 
                    'xgb':s.XGBClassifier(random_state = random_state),
                    'rf_tuned': s.RandomForestClassifier(**s.study(method='rf', n_trials = n_trials, X_train=x_train_select, y_train=y_train_select, X_val=x_val_select, y_val=y_val)),
                    'rf':s.RandomForestClassifier(random_state = random_state),
                    'svm_tuned': s.SVC(**s.study(method='svm', n_trials = n_trials, X_train=x_train_select, y_train=y_train_select, X_val=x_val_select, y_val=y_val)),
                    'svm':s.SVC(random_state = random_state)
                    }
            
            ml_df = s.basic_ml(models, x_train_select, y_train_select, x_test_select, y_test )
            ml_df.insert(0, 'imbalance_method', imb_method_name)
            imb_df = s.pd.concat([imb_df, ml_df])
        
        imb_df = imb_df.reset_index(drop=True)
        imb_df.insert(0, 'feature_selection_method', fs_method_name)
        fs_df = s.pd.concat([fs_df, imb_df])
        
    fs_df = fs_df.reset_index(drop=True)
    fs_df.to_csv(f'{today}_{i+1}/{k_fold_num}_output.csv')
    cv_score.append(fs_df.iloc[:, 3:].values)

final_df = fs_df.copy()
final_df.iloc[:, 3:] = s.np.mean(cv_score, axis = 0)
final_df.to_csv(f'{today}_final_result.csv')
