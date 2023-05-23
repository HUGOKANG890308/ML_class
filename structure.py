from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, fbeta_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold,StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
import scipy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import ExtraTreesClassifier
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

our_random_state = 0

def splitting_train_validation_StratifiedKFold(X, Y, n, our_random_state = None, our_shuffle = False):
    '''
    切出stratifiedkfold的train validation data
    可以用在unbalanced data上 因為在分的時候會讓unbalanced data平均分在每個fold裡
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
    
   
def feature_selection(X, y, method='raw', model=SVC(kernel='rbf', C=10 ),  n_feature=30 , random_state=our_random_state):
    '''
    Input:
    X: input raw data include all feature; type: pandas dataframe
    y: input target: type: pandas dataframe
    method: Method of feature selection, can be one of the following:
            - 'basic_filter_approach': removes constant and duplicate features
            - 'mutual_information_approach': selects features with high mutual information scores
            - 'correlation_filter_approach': removes highly correlated features
            - 'variance_inflation_factor': removes features with high VIF scores
            - 'Sequential Feature Selection': selects features using SFS algorithm
            - 'Exhaustive Feature Selection': selects features using EFS algorithm
            - 'Treebased_embedded_approach': select feature using Tree base algorithm
            - 'raw' : select all feature
    model: Model use in 'Sequential Feature Selection' and 'Exhaustive Feature Selection'; default:SVC(kernel='rbf', C=10)
    n_feature: number of features ; default: 30
    random_state: random state ; type: int; default: 0

    
    Returns:
    pandas dataframe with selected features, X_new & y

    '''
    
    npX = np.array(X)

    if method == 'basic_filter_approach':
        
        threshold = 0 # threshold: float, threshold value for obtain constant feature
        drop_feature = []
        
        vt = VarianceThreshold(threshold=threshold)
        vt.fit(X)
    
        #obtain constant feature
        constant_columns = vt.get_support()
        for i,feature in enumerate(X.columns):
            if constant_columns[i] == False:
                drop_feature.append(feature)
        print('Constant Feature : {}'.format(drop_feature))
        
        #obtain duplicate feature
        X_T = X.T 
        duplicate_feature = X_T[X_T.duplicated()].index.values
        print('Duplicated Features={}'.format(duplicate_feature))
        
        for feature in duplicate_feature:
            drop_feature.append(feature)
            
        # drop feature
        X_new = X.drop(drop_feature, axis=1)
       
        
    elif method == 'mutual_information_approach':
        
        threshold = 0.001 # threshold: float, threshold value for mutual information score
        drop_feature = []
        
        #obtain feature score
        score = mutual_info_classif(X, y, random_state=random_state)
        for i, feature in enumerate(X.columns):
            print('{}={}'.format(feature, score[i]))
            
        #drop feature
        feature_score = dict(zip(X.columns, score))
        sorted_feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
        low_score_features = [f[0] for f in sorted_feature_score if f[1] < threshold]
        X_new = X.drop(low_score_features, axis=1)
        
        
    elif method == 'correlation_filter_approach':
        ''''
        無法列印多feature的correlation matrix 建議不要用，
        用variance_inflation_factor可以得到類似效果
        '''
        corr_method = ['pearson', 'spearman', 'kendall']
        f_size = len(X.columns)
        h_str = '\t'
        for s1 in X.columns:
            h_str = h_str + s1 + '\t'
        for (corr_idx,corr_str) in enumerate(corr_method):
            r_str, p_str = '', ''
            for i1 in range(0, f_size):
                r_str = r_str + X.columns[i1] + '\t'
                p_str = p_str + X.columns[i1] + '\t'
                for i2 in range(0, f_size):
                    if corr_idx==0:  
                        res = scipy.stats.pearsonr(npX[:,i1], npX[:,i2])
                    if corr_idx==1:  
                        res = scipy.stats.spearmanr(npX[:,i1], npX[:,i2])        
                    if corr_idx==2:  
                        res = scipy.stats.kendalltau(npX[:,i1], npX[:,i2])
                    r_str = r_str + '{:.4f}\t'.format(res[0])
                    p_str = p_str + '{:.4f}\t'.format(res[1])
                    r_str = r_str + '\n'
                    p_str = p_str + '\n'

                print('-'*80, '\n{}\'s correlation'.format(corr_str))
                print(h_str,'\n',r_str)

                print('\n{}\'s prob for testing non-correlatio'.format(corr_str))
                print(h_str,'\n',p_str)
        
        
    elif method == 'variance_inflation_factor':
        
        threshold = 10
        f_size = len(X.columns)
        X = X.assign(const=1)
        record = 0
        for i in range(f_size):
            r_str = ''
            for i1 in range(0, f_size - 3 + 1):
                r_str = r_str + X.columns[i1] \
                        + '\t{:.4f}\t'.format(variance_inflation_factor(X.values,i1)) + '\n'
            # print(r_str)
            if variance_inflation_factor(X.values,record) > threshold:
                X.drop([X.columns[record]], axis=1, inplace=True)
                f_size -= 1
            else:
                record += 1
        X_new = X.drop(['const'], axis=1)
        
        
    elif method == 'Sequential Feature Selection':
        
        model = model
        sfs = SFS(model, forward=True, cv=5, floating=False, k_features = n_feature,
                scoring='recall_weighted', verbose=0, n_jobs=-1)
        sfs.fit(X, y)
        # print('Best score achieved:{}, Feature\'s names: {}\n'.format(sfs.k_score_, sfs.k_feature_names_))
        for i1 in tqdm(sfs.subsets_):
            # print('{}\n{}\n'.format(i1, sfs.subsets_[i1]))
            pass            
        # drop feature
        columns = []
        for feature in sfs.k_feature_names_:
            columns.append(feature)
        X_new = X[columns]
            
    elif method == 'Exhaustive Feature Selection':
        '''跑很久'''
        
        model = model
        efs = EFS(model, cv=5, min_features=50, max_features=60, scoring='recall_weighted', n_jobs=-1)
        efs.fit(X, y, custom_feature_names=X.columns.values)
        print('Best score achieved:{}, Feature\'s names: {}\n'.format(efs.best_score_, efs.best_feature_names_))
    
        #drop feature
        X_new = X[sfs.k_feature_names_]      

        
    elif method == 'Treebased_embedded_approach':
        threshold = 0.001 # threshold: float, threshold value for Treebased_embedded_approach - feature_importances
        model = ExtraTreesClassifier(n_estimators=50)
        model.fit(X, y)
        for i, imp in enumerate(model.feature_importances_):
            print('{} = {}'.format(X.columns[i], imp)) 
            
        #drop feature
        feature_importances = model.feature_importances_
        drop_columns = X.columns[feature_importances < threshold]
        X_new = X.drop(drop_columns, axis=1)
        
    
    elif method == 'raw':
        X_new = X
    
    return X_new, y

def imblance_data(X_train, y_train, sample_no, random_state = our_random_state):
    '''
    X_train: input training data ; type: pandas dataframe
    y_train: input training label ; type: pandas dataframe
    sample_no: input input sampling method; type: int
    method: 
            1. ROS
            2. RUS
            3. SMOTE
            4. ADASYN
            5. SMOTETomek
    '''
    if sample_no == 6:
        X_res = X_train
        y_res = y_train
    else:
        if sample_no == 1:
            sample = RandomOverSampler(sampling_strategy='not majority', random_state = random_state)
        elif sample_no == 2:
            sample = RandomUnderSampler(sampling_strategy = 'majority', random_state = random_state)
        elif sample_no == 3:
            sample = SMOTE(sampling_strategy='not majority', random_state = random_state, n_jobs=-1)
        elif sample_no == 4:
            sample = ADASYN(sampling_strategy='not majority', random_state = random_state, n_jobs=-1)
        elif sample_no == 5:
            sample = SMOTEENN(sampling_strategy='not majority', smote=SMOTE(sampling_strategy='not majority', 
                                                                  random_state = random_state, n_jobs=-1))
        
        X_res, y_res = sample.fit_resample(X_train, y_train)     
   
    
        
    '''
    X_res: input training data ; type: pandas dataframe
    y_res: input training label ; type: pandas dataframe
    '''
    return X_res, y_res

def evaluation(y_test, y_pred):
    '''
    to return metrics score
    
    input:
        y_test: input true label; type: pandas dataframe
        y_pred: input prediction; type: pandas dataframe
        
    output:
        evaluation result; type: tuple
    '''
    ac = round(accuracy_score(y_test, y_pred),4)
    f1 = round(f1_score(y_test, y_pred),4)
    pre = round(precision_score(y_test, y_pred),4)
    rec = round(recall_score(y_test, y_pred),4)
    auc =round(roc_auc_score(y_test, y_pred),4)
    f_beta = round(fbeta_score(y_test, y_pred, beta=3),4)
    
    return ac, f1, pre, rec, auc, f_beta
def basic_ml(using_model , X_train, y_train, X_test, y_test ):
    '''
    to return evaluate dataframe
    
    input:
        using_model: input using model; type: dictionary
        x_train: input x_train; type: numpy.ndarray or dataframe
        y_train: input y_train; type: numpy.ndarray or dataframe
        x_test: input x_test; type: numpy.ndarray  or dataframe
        y_test: input y_test; type: numpy.ndarray  or dataframe
    
    output:
        evaluate dataframe
    '''
    score = []
    for i in using_model:
        model = using_model[i]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score.append([i]+list(evaluation(y_test, y_pred)))
    return pd.DataFrame(data = score, columns = ['model', 'accuracy', 
                                                 'f1_score', 'precision', 'recall', 'auc', 'f_beta'])


# example of using basic_ml
'''
df = basic_ml(using_model={'xgb': XGBClassifier(), 'rf': RandomForestClassifier(
)}, X_train, y_train, X_test, y_test)
'''

    
def objective(trial, method, X_train, y_train, X_val, y_val):
    '''
    method: input using model; type: string
         can be one of the following: 'svm', 'rf', 'xgb'
    clf: input using model; type: sklearn model
    '''
    if method == 'svm':
        C = trial.suggest_loguniform('C', 1e-5, 1e5)
        kernel = trial.suggest_categorical(
            'kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        degree = trial.suggest_int('degree', 2, 5)
        clf=SVC(C=C, kernel=kernel, degree=degree,random_state=our_random_state)

    elif method == 'rf':
        max_depth = trial.suggest_int("max_depth", 2, 128)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 128)
        max_leaf_nodes = int(trial.suggest_int("max_leaf_nodes", 2, 128))
        min_samples_leaf = int(trial.suggest_int('min_samples_leaf', 2, 128))
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        clf=RandomForestClassifier(min_samples_split=min_samples_split,
                max_leaf_nodes=max_leaf_nodes, criterion=criterion, random_state=our_random_state, max_depth=max_depth,
                min_samples_leaf=min_samples_leaf)
    elif method == 'xgb':
        max_depth = trial.suggest_int("max_depth", 2, 128)
        min_child_weight = trial.suggest_int("min_child_weight", 2, 128)
        gamma = trial.suggest_int("gamma", 2, 128)
        subsample = trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.1)
        colsample_bytree = trial.suggest_discrete_uniform(
            'colsample_bytree', 0.5, 1, 0.1)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        clf=XGBClassifier(max_depth=max_depth, min_child_weight=min_child_weight, gamma=gamma, subsample=subsample,
                colsample_bytree=colsample_bytree, learning_rate=learning_rate,
                random_state=our_random_state)#, tree_method='gpu_hist' if torch.cuda.is_available() else 'auto')
    else:
        raise ValueError(f"Invalid method '{method}'")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    scores =fbeta_score(y_val, y_pred, beta=3)
    '''
    output:f_beta score
    '''
    return scores

def study(method, n_trials,X_train, y_train, X_val, y_val):
    '''
    method : input using model; type: string, 
        can be one of the following: 'svm', 'rf', 'xgb'
    n_trials : input number of trials; type: int
    X_train: input training data ; type: pandas dataframe
    y_train: input training label ; type: pandas dataframe
    X_val: input validation data ; type: pandas dataframe
    y_val: input validation label ; type: pandas dataframe
    
    '''
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, method,X_train, y_train, X_val, y_val), n_trials=n_trials)
    '''
    output: best params of model type: dictionary
    '''
    return study.best_params



'''
example of using study
basic_ml(using_model={'xgb': XGBClassifier(**study(method='xgb', n_trials=10 ,X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)), 
                      'xgb1':XGBClassifier(random_state=our_random_state),
                      'rf': RandomForestClassifier(**study(method='rf', n_trials=10 ,X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)),
                      'rf1':RandomForestClassifier(random_state=our_random_state),
                      'svm': SVC(**study(method='svm', n_trials=10 ,X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)),
                      'svm1':SVC(random_state=our_random_state),
                      },
basic_ml(using_model={'xgb': XGBClassifier(**study(method='xgb', n_trials=10 ,X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)), 
                      'xgb1':XGBClassifier(random_state=our_random_state),
                      'rf': RandomForestClassifier(**study(method='rf', n_trials=10 ,X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)),
                      'rf1':RandomForestClassifier(random_state=our_random_state),
                      'svm': SVC(**study(method='svm', n_trials=10 ,X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)),
                      'svm1':SVC(random_state=our_random_state),
                      },
 X_train=pd.concat([X_train, X_val], axis=0), y_train=pd.concat([y_train, y_val], axis=0), 
 X_test=X_test, y_test=y_test)

'''

'''
Deeplearning model
使用方法 : 
Step1 : 先將X_train, y_train, X_valid, y_valid, X_test, y_test, 帶入Function "convert_to_DataLoader"
        返還 (train_loader, valid_loader, test_loader)
Step2 : 將train_loader, valid_loader, test_loader帶入Training_nn進行超參數的調教
        返還 Function "evaluation" 中的所有metric
'''
class nn_model(nn.Module):
    '''
    此為deep learning model 
    可調教的參數有 n_layers, hidden_sizes, learning_rate
    '''
    def __init__(self, input_size, hidden_sizes, output_size):
        super(nn_model, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.Sigmoid())
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

    
def convert_to_DataLoader(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size):
    '''
    這個function用來將training validation, testing data 轉換為 Pytorch DataLoader
    input:
        X_train, y_train, X_valid, y_valid, X_test, y_test : 
            這些data 必須經過imbalance data的處理以及標準化
        batch_size : 
            用來決定DataLoader 一次迭帶多少筆data
    output:
        train_loader, valid_loader, test_loader :
            Pytorch DataLoader
    '''
    shuffle = False
    batch_size = batch_size
    # Convert data to pytorch DataLoader 
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    
    #Create train dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    #Create validation dataloader
    valid_dataset = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

    #Create train dataloader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle) 
    
    return train_loader, valid_loader, test_loader

def objective_nn(trial, train_loader, valid_loader, input_size):
    '''
    optuna_object for nn_model, hyperparameter(n_layers, hidden_sizes, learning_rate)

    input : train_loader, valid_loader :
            
    output : our main metric
    '''
    # 定義參數
    input_size = input_size
    n_epochs = 20
    output_size = 1
    print("*"*50)
    hidden_sizes = []
    n_layers = trial.suggest_int("n_layers", 1, 5)  # Suggest the number of layers to be tuned
    for i in range(n_layers):
        hidden_sizes.append(trial.suggest_int(f"hidden_size_{i}", 2, 64))  # Suggest the size of each hidden layer
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    
    
    model = nn_model(input_size, hidden_sizes, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.eval()
    
    predictions = []
    y_valids = []
    score = None
    with torch.no_grad():
        for x_valid, y_valid in valid_loader:
            outputs = model(x_valid)
            predictions.extend(outputs.round().squeeze().tolist())
            y_valids.extend(y_valid.round().squeeze().tolist())
    # Calculate F1 score
    score = fbeta_score(y_valids, predictions, beta=3) 
    #score = accuracy_score(y_valids, predictions)
    return score

def Training_nn(train_loader, valid_loader, test_loader, input_size, n_epochs, batch_size, n_trials):
    '''
    這個Function用來訓練及調教nn_model中的超參數
    input :
        train_loader, valid_loader, test_loader
        input_size : number of feature after feature selection 
        n_epochs : number od epoch
        batch_size : number of data for every iteration
        n_trials : number of trials for optuna
    output :
        Our evaluation metrics
        ac, f1, pre, rec, auc, f_beta
    '''
    # 定義引數
    hidden_size = []
    output_size = 1
    learning_rate = None
    
    n_trials = n_trials
    batch_size = batch_size
    n_epochs = n_epochs
    shuffle = False
    

    
    # Hyperparametor tuneing using optuna
    
    ## 設定 sampler
    #alg=optuna.samplers.GridSampler(seed=42)   # Grid search
    #alg=optuna.samplers.RandomSampler(seed=42) # Random search
    alg=optuna.samplers.TPESampler(seed=42) # Tree-structured Parzen Estimator algorithm.   
    #alg=optuna.samplers.CmaEsSampler(seed=42) # CMA-ES based algorithm
    #alg=optuna.samplers.NSGAIISampler(seed=42)# Nondominated Sorting Genetic Algorithm II 
    #alg=optuna.samplers.MOTPESampler(seed=42) # Multiobjective Tree-structured Parzen Estimator
    #alg=optuna.samplers.QMCSampler(seed=42) # A Quasi Monte Carlo sampling algorithm
    #alg=optuna.samplers.BruteForceSampler(seed=42) # Brute force algorithm.
    #alg=optuna.samplers.intersection_search_space()# the intersection of parameter distributions that have been suggested in the completed trials of the study so far.
    #alg=optuna.samplers.IntersectionSearchSpace()  # provides the same functionality of intersection_search_space with a much faster way
    
    
    ## 設定 pruners (修剪方法)
    # pruner=optuna.pruners.MedianPruner()     # Prune with median. For RandomSampler, MedianPruner
    # pruner=optuna.pruners.NopPruner()        # No pruning
    # pruner=optuna.pruners.PatientPruner()    # Prune with tolerance
    # pruner=optuna.pruners.PercentilePruner() # Prune with specified percentile
    # pruner=optuna.pruners.SuccessiveHalvingPruner() # Prune with Asynchronous Successive Halving algorithm
    pruner=optuna.pruners.HyperbandPruner() # SuccessiveHalvingPruner的更激進版本, 並根據中間結果修剪試驗. For TPESampler, HyperbandPruner is the best.
    # pruner=optuna.pruners.ThresholdPruner() # 當目標函數的值超過或是低於給定閾值時，該剪枝器停止試驗
    
    ## tuneing model 
    study = optuna.create_study(sampler=alg, direction='maximize')
    study.optimize(lambda trial: objective_nn(trial,  train_loader=train_loader, valid_loader=valid_loader, input_size=input_size), n_trials=n_trials)
    best_params =  study.best_params
        
    # Retrain model
    hidden_sizes = list(best_params.values())[1:-2]
    n_layers = list(best_params.values())[0]
    learning_rate = list(best_params.values())[-1]
    best_model = nn_model(input_size, hidden_size, output_size)
    best_optimizer = optim.Adam(best_model.parameters(), lr=learning_rate)  # Define the optimizer
    criterion = torch.nn.BCELoss()
    
    best_model.train()
    for epoch in range(n_epochs):
        for X_train, y_train in train_loader:
            best_optimizer.zero_grad()  # Use the best optimizer
            outputs = best_model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1))
            loss.backward()
            best_optimizer.step()
            
    # Evaluate the best model on the test set
    best_model.eval()
    predictions = []
    y_tests = []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            outputs = best_model(X_test)
            predictions.extend(outputs.round().squeeze().tolist())
            y_tests.extend(y_test.round().squeeze().tolist())
            
    # Calculate F1 score on the test set
    test_score = fbeta_score(y_tests, predictions, beta=3)
    print("Best Fbata_score on the test set: {:.4f}".format(test_score))
    
    score = []
    score.append(['NN']+list(evaluation(y_tests, predictions)))
    return pd.DataFrame(data = score, columns = ['model','accuracy', 'f1_score', 'precision',
                                                 'recall', 'auc', 'f_beta'])
     
'''
Deep Learning example
# Difine parameter
n_epochs = 10
batch_size = 128
n_trials = 100
input_size = X_train.shape[1]
random_state = 0
shuffle = False

train_loader, valid_loader, test_loader = convert_to_DataLoader(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size)
ac, f1, pre, rec, auc, f_beta = Training_nn(train_loader, valid_loader, test_loader, input_size, n_epochs, batch_size, n_trials)

'''
    

if __name__ == '__main__':
    '''
    main function
    write your code here
    ex:
    using_model={'DecisionTreeClassifier':DecisionTreeClassifier()}
    '''
    '''
    using_model={'DecisionTreeClassifier':DecisionTreeClassifier()}
    for i in using_model:
        print(i)
        evaluate_classifier(using_model[i],X_train, y_train, X_test, y_test)
        print('------------------')
    '''
   
