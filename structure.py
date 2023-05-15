from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, fbeta_score
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

def train_test_spiltting(data,test_size=0.2,random_state=0):
    '''
    data: input raw data ; type: pandas dataframe
    test_size: size of training data ; type: float; default: 0.2
    random_state: random state ; type: int; default: 0
    '''
    '''
    train_data: training data ; type: pandas dataframe
    test_data: testing data ; type: pandas dataframe
    '''
    return train_data,test_data
    
def train_val_spiltting(train_data,val_size=0.2,random_state=0):
    '''
    train_data: input raw data ; type: pandas dataframe
    val_size: size of validation data ; type: float; default: 0.2
    random_state: random state ; type: int; default: 0
    '''
    '''
    train_data: training data ; type: pandas dataframe
    val_data: validation data ; type: pandas dataframe
    '''
    return train_data,val_data

    

def standardize(training_data,validation_data,testing_data,method='what method you use'):
    '''
    training_data: input training data(data after spiltting) ; type: pandas dataframe
    validation_data: input training data(data after spiltting) ; type: pandas dataframe
    testing_data: input training data(data after spiltting) ; type: pandas dataframe
    method: input method you use to standardize data ; type: string
            1. standardize the data
            2. minmax the data
    '''

    '''
    training_data: training data after standardize ; type: pandas dataframe
    validation_data: validation data after standardize ; type: pandas dataframe
    testing_data: testing data after standardize ; type: pandas dataframe
    '''
    return training_data,validation_data,testing_data
   
def feature_selection(training_data,validation_data,testing_data,method='what method you use'):
    '''
    training_data: input training data(data after standardize) ; type: pandas dataframe
    validation_data: input training data(data after standardize) ; type: pandas dataframe
    testing_data: input training data(data after standardize) ; type: pandas dataframe
    method: 
            1. drop features by domain knowledge
            2. wrapper method or filter method or embedded method
    '''
    '''
    training_data: training data after feature selection ; type: pandas dataframe
    validation_data: validation data after feature selection ; type: pandas dataframe
    testing_data: testing data after feature selection ; type: pandas dataframe
    '''
    return training_data,validation_data,testing_data

def imblance_data(training_data,validation_data,testing_data,method='what method you use'):
    '''
    training_data: input training data(data after feature selection) ; type: pandas dataframe
    validation_data: input training data(data after feature selection) ; type: pandas dataframe
    testing_data: input training data(data after feature selection) ; type: pandas dataframe
    method: 
            1. over sampling
            2. under sampling
            3. SMOTE
            4. ADASYN
    '''
    '''
    training_data: training data after imblance ; type: pandas dataframe
    validation_data: validation data after imblance ; type: pandas dataframe
    testing_data: testing data after imblance ; type: pandas dataframe
    '''
    return training_data,validation_data,testing_data

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

def basic_ml(using_model = dict, X_train, y_train, X_test, y_test ):
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

def evaluation(y_test, y_pred):
    '''
    to return metrics score

    input:
        y_test: input true label; type: pandas dataframe
        y_pred: input prediction; type: pandas dataframe

    output:
        evaluation result; type: tuple
    '''
    ac = round(accuracy_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)
    pre = round(precision_score(y_test, y_pred), 4)
    rec = round(recall_score(y_test, y_pred), 4)
    auc = round(roc_auc_score(y_test, y_pred), 4)
    f_beta = round(fbeta_score(y_test, y_pred, beta=3), 4)

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

# example of using basic_ml
'''
df = basic_ml(using_model={'xgb': XGBClassifier(), 'rf': RandomForestClassifier(
)}, X_train, y_train, X_test, y_test)
'''

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
'''
example of using optuna
basic_ml(using_model={'xgb': XGBClassifier(**study(method='rf', n_trials=10)),'xgb1':XGBClassifier()},
 X_train=pd.concat([X_train, X_val], axis=0), y_train=pd.concat([y_train, y_val], axis=0), 
 X_test=X_test, y_test=y_test)

'''


def deep_learning_model():
    '''
    deep learning model,
    how to do please think by yourself
    
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
   