import structure as s
random_state = 0
test_size = 0.2
k_fold_num = 5
fs_model = s.SVC(kernel='rbf', C=10 )
fs_feature_num = 30
n_trials = 10

df = s.pd.read_csv('data.csv')
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']

# train, test split
x_train, x_test, y_train, y_test = s.train_test_split(X, Y, test_size = test_size, random_state = random_state)
x_train_select, y_train_select = s.feature_selection(x_train, y_train, 'variance_inflation_factor', fs_model, fs_feature_num, random_state)
x_test_select = x_test[x_train_select.columns] ; x_val_select = x_val[x_train_select.columns]
        
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
    cv_score.append(fs_df.iloc[:, 3:].values)

final_df = fs_df.copy()
final_df.iloc[:, 3:] = s.np.mean(cv_score, axis = 0)
final_df.to_csv('final_result.csv')
