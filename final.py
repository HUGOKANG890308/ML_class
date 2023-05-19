'''
    所有 {}_train, {}_valid, {}_test 的 x,y 都是小寫開頭
    變數(Test_size, Valid_size.....)都是大寫開頭
    line 18: 用在 Sequential Feature Selection
    line 21~25: 讓 for 迴圈跑的 list，line 25應該是 dict(用在 basic_ml)，但是不確定該怎麼寫
    line 46: 有兩個 split train & valid 的方法，但當初講迴圈的時候後沒有提到，不確定是否該用迴圈寫這部分
    line 82: 不清楚 x_train 和 y_train 的輸入應該怎麼處理
    line 88: 存成csv檔，但目前每一項都會存成一樣的檔名
'''

import structure.py as s

Test_size = 0.2
Valid_size = 0.2
Random_state = 0
N_split = 5
Shuffle = False
Model = s.SVC(kernel='rbf', C=10 )
N_feature = 5

list_train_val_split = ['train_test_split', 'StratifiedKFold']
list_standardize = ['z_score_normalization', 'min_max']
list_feature_selection = ['variance_inflation_factor', 'Sequential Feature Selection', 'raw']
list_oversample = ['1', '2', '3', '4', '5', '6']
using_model = {'xgb': s.XGBClassifier(**s.study(method='xgb', n_trials=10)),'xgb1':s.XGBClassifier()}



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
for method_split in list_train_val_split:
    if method_split == 'train_test_split':
        x_train, x_valid, y_train, y_valid = s.train_test_split(x_train, y_train, 
                                                                test_size = Valid_size, 
                                                                random_state = Random_state)
    else:
        x_train, x_valid, y_train, y_valid = s.splitting_train_validation_StratifiedKFold(x_train, y_train, 
                                                                                          N_split, 
                                                                                          Random_state, 
                                                                                          Shuffle)
    
    # standardize train & test
    for method_standardize in list_standardize:
        x_train, x_valid, x_test = s.standardize(x_train, x_valid, x_test, method_standardize)
        
        
        
        # feature selection
        for method_feature in list_feature_selection:
            if method_feature == 'Sequential Feature Selection':
                x_train, y_train = s.feature_selection(x_train, y_train, 'variance_inflation_factor', 
                                                       Model, N_feature, Random_state)
            
            x_train, y_train = s.feature_selection(x_train, y_train, method_feature, Model, N_feature, Random_state)
            # print(f'x_train is {x_train}, x_train.shape is {x_train.shape} \n\n')
            
            
            
            # unbalanced data processing
            for method_oversample in list_oversample:
                x_train, y_train = s.imblance_data(x_train, y_train, method_oversample, Random_state)
                
                
                
                # model training
                basic_ml(using_model, 
                         X_train=pd.concat([X_train, X_val], axis=0), y_train=pd.concat([y_train, y_val], axis=0), 
                         x_test, y_test)
                
                
                
                # Save as a clean file
                df.to_csv('ML.csv', index=False)
                
       