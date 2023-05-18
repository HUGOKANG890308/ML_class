import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import RandomOverSampler, RandomUnderSampler, SMOTE, ADASYN
from imblearn.combine import SMOTEENN

from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
import scipy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import ExtraTreesClassifier

import optuna

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, fbeta_score



def splitting_train_validation_StratifiedKFold(X, Y, n, our_random_state = None, our_shuffle = False):
    
    
    '''
    切出 stratifiedkfold 的 train validation data
    可以用在 unbalanced data 上 因為在分的時候會讓 unbalanced data 平均分在每個 fold 裡
    input:
        X = x_train ; type = pandas dataframe
        Y = y_train ; type = pandas dataframe
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
                1. z score normalize the data
                2. minmax the data

    output:
        x_training_data = training data after standardize ; type = pandas dataframe
        x_validation_data = validation data after standardize ; type = pandas dataframe
        x_testing_data = testing data after standardize ; type = pandas dataframe
    '''

    if method == 'z_score_normalization':
        temp = StandardScaler()
        
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
def imblance_data(X_train, y_train, sample_no, random_state):
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
def feature_selection(X, y, method='raw', model=SVC(kernel='rbf', C=10 ),  n_feature=30 , random_state=0):
    
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
            print(r_str)
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
        sfs.fit(X, y, custom_feature_names=X.columns.values)
        print('Best score achieved:{}, Feature\'s names: {}\n'.format(sfs.k_score_, sfs.k_feature_names_))
        for i1 in sfs.subsets_:
            print('{}\n{}\n'.format(i1, sfs.subsets_[i1]))
        
        # drop feature
        X_new = X[sfs.k_feature_names_]
            
            
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

def objective(trial, method='svm'):
    if method == 'svm':
        C = trial.suggest_loguniform('C', 1e-5, 1e5)
        kernel = trial.suggest_categorical(
            'kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        degree = trial.suggest_int('degree', 2, 5)
        clf = SVC(C=C, kernel=kernel, degree=degree)
        clf.fit(X_train, y_train)
    elif method == 'rf':
        max_depth = trial.suggest_int("max_depth", 2, 128)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 128)
        max_leaf_nodes = int(trial.suggest_int("max_leaf_nodes", 2, 128))
        min_samples_leaf = int(trial.suggest_int('min_samples_leaf', 2, 128))
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        clf = RandomForestClassifier(min_samples_split=min_samples_split,
                                     max_leaf_nodes=max_leaf_nodes,
                                     criterion=criterion, random_state=4,
                                     max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        clf.fit(X_train, y_train)
    elif method == 'xgb':
        max_depth = trial.suggest_int("max_depth", 2, 128)
        min_child_weight = trial.suggest_int("min_child_weight", 2, 128)
        gamma = trial.suggest_int("gamma", 2, 128)
        subsample = trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.1)
        colsample_bytree = trial.suggest_discrete_uniform(
            'colsample_bytree', 0.5, 1, 0.1)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

        clf = XGBClassifier(max_depth=max_depth, min_child_weight=min_child_weight, gamma=gamma, subsample=subsample,
                            colsample_bytree=colsample_bytree, learning_rate=learning_rate)
        clf.fit(X_train, y_train)
    else:
        raise ValueError(f"Invalid method '{method}'")
    y_pred = clf.predict(X_val)
    scores =fbeta_score(y_val, y_pred, beta=3)
    return scores
def study(method='xgb', n_trials=10):
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, method='xgb'), n_trials=n_trials)
    return study.best_params
X_train, y_train, X_test, y_test = None
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
def basic_ml(using_model={'xgb': XGBClassifier(), 'rf': RandomForestClassifier()},
             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
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
    return pd.DataFrame(data=score, columns=['model', 'accuracy', 'f1_score', 'precision', 'recall', 'auc', 'f_beta'])



df = pd.read_csv('data.csv')
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']
# print(f'X is {X}, X.shape is {X.shape} /n/n')
# print(f'Y is {Y}, Y.shape is {Y.shape} /n/n')



# print('split train & test...... /n')
Test_size = input('please input the test_size: ')
Random_state = input('please input the random_state: ')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = Test_size, 
                                                    random_state = Random_state)



# print('split train & valid...... /n')
Way = input('please input the way to split the train & valid data: /n'
      '    1. train_test_split() /n'
      '    2. StratifiedKFold /n'
      '    (train_test_split / StratifiedKFold): ')

x_validation, y_validation = None

if Way == 'train_test_split':    
    Validation_size = input('please input the test_size: ')
    Validation_state = input('please input the random_state: ')
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = Validation_size, 
                                                                    random_state = Validation_state)
elif Way == 'StratifiedKFold':
    N_split = input('please input the n_split: ')
    SKFold_random_state = input('please input the random_state: ')
    SKFold_shuffle = input('please input the shufle: ')
    x_train, x_validation, y_train, y_validation = splitting_train_validation_StratifiedKFold(x_train, y_train, N_split, 
                                                                        SKFold_random_state, SKFold_shuffle)
else:
    print('wrong input of splitting train & valid way /n/n')
    exit



# print('standardize train & test...... /n')
Way = input('please input the way to standardize the train & valid data: /n'
            '    1. z_score_normalization /n'
            '    2. min_max /n'
            '    (z_score_normalization / min_max): ')

x_train, x_validation, x_test = standardize(x_train, x_validation, x_test, Way)



# print('unbalanced data processing...... /n')
Way = input('please input the way to process the unbalanced data: /n'
            '    1. ROS /n'
            '    2. RUS /n'
            '    3. SMOTE /n'
            '    4. ADASYN /n'
            '    5. SMOTETomek /n'
            '    6. raw /n'
            '    (1 / 2 / 3 / 4 / 5 / 6): ')
Random_state = input('please input the random_state: ')

x_train, y_train = imblance_data(x_train, y_train, Way, Random_state)

    

# print('feature selection...... /n')
Way = input('please input the way to process the feature selection: /n'
            '    1. basic_filter_approach /n'
            '    2. mutual_information_approach /n'
            '    3. correlation_filter_approach /n'
            '    4. variance_inflation_factor /n'
            '    5. Sequential Feature Selection /n'
            '    6. Exhaustive Feature Selection /n'
            '    7. Treebased_embedded_approach /n'
            '    8. raw /n'
            '    (basic_filter_approach / mutual_information_approach '
                '/ correlation_filter_approach / variance_inflation_factor '
                '/ Sequential Feature Selection / Exhaustive Feature Selection '
                '/ Treebased_embedded_approach / raw): ')

if Way == 'Sequential Feature Selection' or Way == 'Exhaustive Feature Selection':
    Model = input('please input the model: ')
else:
    Model = None
N_feature = input('please input the n_feature: ')
Random_state = input('please input the random_state: ')

X, Y = feature_selection(X, Y, Way, Model,  N_feature, Random_state)
# print(f'X is {X}, X.shape is {X.shape} /n/n')
# print(f'Y is {Y}, Y.shape is {Y.shape} /n/n')



run_loop = True
while run_loop == True:
    
    # print('choosing best parameters...... /n')
    Way = input('please input the way to process the parameters choosing: /n'
                '    1. svm /n'
                '    2. rf /n'
                '    3. xgb /n'
                '    (svm / rf / sgb): ')
    N_trials = input('please input the n_trials: ')
    Best_params = study(Way, n_trials=10)



    # print('model training...... /n')
    using_model = {'xgb': XGBClassifier(), 'rf': RandomForestClassifier()}
    df = basic_ml(using_model, x_train, y_train, x_test, y_test)
    # print(df)
    
    
    
    run_loop = input('continue running the loop?(True / False): ')
    print('----------' * 4)
    








