from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, fbeta_score
import pandas as pd
import numpy as np
def evaluation(y_test, y_pred):
    '''
    to return metrics score
    
    input:
        y_test: input true label; type: pandas dataframe
        y_pred: input prediction; type: pandas dataframe
        
    output:
        evaluation result; type: tuple
    '''
    ac = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f_beta = fbeta_score(y_test, y_pred, beta=3)
    
    return ac, f1, pre, rec, auc, f_beta

def basic_ml(using_model = {'xgb': XGBClassifier(),'rf': RandomForestClassifier()},
             X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test):
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
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score.append([i]+list(evaluation(y_test, y_pred)))
        
    return pd.DataFrame(data = score, columns = ['model', 'accuracy', 'f1_score', 'precision', 'recall', 'auc', 'f_beta'])