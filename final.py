'''
    所有 {}_train, {}_valid, {}_test 的 x,y 都是小寫開頭
    變數(Test_size, Valid_size.....)都是大寫開頭
    line 77: 存成csv檔，但目前每一項都會存成一樣的檔名，且沒有存到 object
'''

import structure as s

Test_size = 0.2
Valid_size = 0.2
Random_state = 0
N_split = 5
Shuffle = False
Model = s.SVC(kernel='rbf', C=10 ) # 用在 Sequential Feature Selection
N_feature = 5
N_trials = 10

method_standardize = 'min_max' # 有兩個 split train & valid 的方法: z_score_normalization / min_max
list_feature_selection = ['variance_inflation_factor', 'Sequential Feature Selection', 'raw']
list_oversample = ['1', '2', '3', '4', '5', '6']
using_model = {'xgb_tuned': s.XGBClassifier(**s.study(method='xgb', n_trials=N_trials, 
                                                      X_train=x_train, y_train=y_train, 
                                                      X_val=x_valid, y_val=y_valid)),
               'xgb': s.XGBClassifier(random_state=Random_state),
               
               'rf_tuned': s.RandomForestClassifier(**s.study(method='rf', n_trials=N_trials, 
                                                              X_train=x_train, y_train=y_train, 
                                                              X_val=x_valid, y_val=y_valid)),
               'rf': s.RandomForestClassifier(random_state=Random_state),
               
               'svm_tuned': s.SVC(**s.study(method='svm', n_trials=N_trials, 
                                            X_train=x_train, y_train=y_train, 
                                            X_val=x_valid, y_val=y_valid)),
               'svm': s.SVC(random_state=Random_state)
               }


# 讀檔
df = s.pd.read_csv('data.csv')
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']
# print(f'X is {X}, X.shape is {X.shape} \n')
# print(f'Y is {Y}, Y.shape is {Y.shape} \n\n')


# split train & test
x_train, x_test, y_train, y_test = s.train_test_split(X, Y, test_size = Test_size, 
                                                    random_state = Random_state)
# print(f'x_train is {x_train}, x_train.shape is {x_train.shape} \n')
# print(f'y_train is {y_train}, y_train.shape is {y_train.shape} \n\n')


# split train & valid
x_train, x_valid, y_train, y_valid = s.splitting_train_validation_StratifiedKFold(x_train, y_train,  
                                                                                  N_split, Random_state,  
                                                                                  Shuffle)
    
# standardize train & test
x_train, x_valid, x_test = s.standardize(x_train, x_valid, x_test, method_standardize)
x_train = s.pd.DataFrame(x_train, columns = X.columns)
x_valid = s.pd.DataFrame(x_valid, columns = X.columns)
x_test = s.pd.DataFrame(x_test, columns = X.columns)   
        
        
# feature selection
for method_feature in list_feature_selection:
    if method_feature == 'Sequential Feature Selection':
        x_train, y_train = s.feature_selection(x_train, y_train, 'variance_inflation_factor', 
                                               Model, N_feature, Random_state)
            
    x_train, y_train = s.feature_selection(x_train, y_train, method_feature, Model, N_feature, Random_state)
    x_valid = x_valid[x_train.columns]
    x_test = x_test[x_train.columns]
    # print(f'x_train is {x_train}, x_train.shape is {x_train.shape} \n\n')
                 
            
    # unbalanced data processing
    for method_oversample in list_oversample:
        x_train, y_train = s.imblance_data(x_train, y_train, method_oversample, Random_state)
                
                    
        # model training
        dataframe = s.basic_ml(using_model, x_train, y_train, x_test, y_test)
        
                
        # Save as a clean file
        dataframe.to_csv('ML_result.csv', index=False)
                
       