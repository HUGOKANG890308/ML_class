import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import optuna

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



def evaluation(y_pred,y_true):
    '''
    input prediction and true label
    y_pred: input prediction ; type: pandas dataframe
    y_true: input true label ; type: pandas dataframe
    '''
    return evaluation_result
    '''
    output evaluation result;type:dataframe
    '''

def default_classifier(clf, X_train, y_train,X_vaild,y_vaild, X_test, y_test):
    '''
    clf: input classifier ; type: sklearn classifier
        XGBoost, LightGBM, CatBoost, RandomForest,SVM, KNN
    X_train: input training data ; type: pandas dataframe
    y_train: input training label ; type: pandas dataframe
    X_vaild: input validation data ; type: pandas dataframe
    y_vaild: input validation label ; type: pandas dataframe
    X_test: input testing data ; type: pandas dataframe
    y_test: input testing label ; type: pandas dataframe
    '''
    evaluation(y_pred,y_true)

class Classifier(object):
    def __init__(self,clf, X_train, y_train,X_vaild,y_vaild, X_test, y_test):
        '''
        clf: input classifier ; type: sklearn classifier
            XGBoost, LightGBM, CatBoost, RandomForest,SVM, KNN
        X_train: input training data ; type: pandas dataframe
        y_train: input training label ; type: pandas dataframe
        X_vaild: input validation data ; type: pandas dataframe
        y_vaild: input validation label ; type: pandas dataframe
        X_test: input testing data ; type: pandas dataframe
        y_test: input testing label ; type: pandas dataframe
        '''
        self.clf=clf
        self.X_train=X_train
        self.y_train=y_train
        self.X_vaild=X_vaild
        self.y_vaild=y_vaild
        self.X_test=X_test
        self.y_test=y_test
    def search_best_params(self):
        '''
        search best params
        '''
        return self.best_params
    def objective(self,trial,params):
        '''
        objective function
        '''
        return self.best_score
    
    def train(self):
        '''
        train model
        '''
        return self.clf
    def predict(self):
        '''
        predict result
        '''
        return self.y_pred
    def evaluate(self):
        '''
        evaluate model
        '''
        return self.evaluation_result
    def save_model(self):
        '''
        save model
        '''
        return self.model
    def main(self):
        '''
        main function
        '''
        self.best_params=self.search_best_params()
        self.clf=self.train()
        self.y_pred=self.predict()
        self.evaluation_result=self.evaluate()
        self.model=self.save_model()
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
       