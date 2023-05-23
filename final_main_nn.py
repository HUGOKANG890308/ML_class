import structure as s

# some cofig

random_state = 0
test_size = 0.2
k_fold_num = 5
fs_feature_num = 30
fs_model = s.SVC(kernel='rbf', C=10 )
n_trials = 100
n_epochs = 10
batch_size = 128

# Tune best feature_selection and imbaloance_data method

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

df = s.pd.read_csv('data.csv')
X = df.drop(['Bankrupt?'], axis=1)
y = df['Bankrupt?']

x_train, x_test, y_train, y_test = s.train_test_split(X, y, test_size=0.2, random_state=random_state)
x_train, x_val, y_train, y_val = s.splitting_train_validation_StratifiedKFold(x_train, y_train, 5)
x_train, x_val, x_test = s.standardize(x_train, x_val, x_test, 'min_max')
x_train = s.pd.DataFrame(x_train, columns = X.columns)
x_val = s.pd.DataFrame(x_val, columns = X.columns)
x_test = s.pd.DataFrame(x_test, columns = X.columns)

# feature selection
final_df = s.pd.DataFrame()
for fs_method_name, fs_method in s.tqdm(feature_selection_mtehod.items()):
    if fs_method_name == 'VIF + SFS':
        x_train, y_train = s.feature_selection(x_train, y_train, 'variance_inflation_factor', random_state)
    
    x_train, y_train = s.feature_selection(x_train, y_train, fs_method, fs_model, fs_feature_num, random_state)
    x_test = x_test[x_train.columns] ; x_val = x_val[x_train.columns]
    
    # Define input_size for nn_model
    input_size = len(x_train.columns)
    
    # training data Imbalance process
    imb_df = s.pd.DataFrame()
    for imb_method_name, method_no in imbalance_method.items():
        x_train, y_train = s.imblance_data(x_train, y_train, method_no, random_state)
    
        # Traning nn_model
        X_train = s.np.array(x_train)
        X_valid = s.np.array(x_val)
        X_test = s.np.array(x_test)
        y_train = s.np.array(y_train)
        y_valid = s.np.array(y_val)
        y_test = s.np.array(y_test)


        train_loader, valid_loader, test_loader = s.convert_to_DataLoader(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size)
        ml_df = s.Training_nn(train_loader, valid_loader, test_loader, input_size, n_epochs, batch_size, n_trials)
        ml_df.insert(0, 'imbalance_method', imb_method_name)
        imb_df = s.pd.concat([imb_df, ml_df])
    #重置索引，否则会出现重复的索引，
    #當 inplace=True 時，DataFrame 會直接被修改，並且該方法不會返回新的 DataFrame。
    # 如果 inplace=False（默認值），該方法會返回一個修改後的 DataFrame 的副本，並且原始 DataFrame 保持不變。
    imb_df = imb_df.reset_index(drop=True)
    imb_df.insert(0, 'feature_selection_method', fs_method_name)
    final_df = s.pd.concat([final_df, imb_df])
final_df = final_df.reset_index(drop=True)
final_df.to_csv('final_result.csv')
