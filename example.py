import pandas as pd
import numpy as np
class data_process():
    def __init__(self, data):
        '''
        input tpye: pandas.DataFrame
        '''
        self.data = data
    def normalize(self):
        '''
        normalize the data
        method: 
            1.min-max normalization, x = (x - min) / (max - min)
            2.standardization, x = (x - mean) / std
        '''
        return self.data
        '''
        output type: pandas.DataFrame
        '''
    def feature_selection(self):
        '''
        feature selection
        method:
            1.pearson correlation
            2.lasso
            3.random forest
        '''
        return self.data
        '''
        output type: pandas.DataFrame
        '''
    def feature_engineering(self):
        '''
        feature engineering
        method:
            1.pca
            2.svd
            3.random projection
            https://scikit-learn.org/stable/modules/random_projection.html
        '''
        return self.data
        '''
        output type: pandas.DataFrame
        '''
    def main(self, method=['normalize', 'feature_selection', 'feature_engineering']):
        '''
        main function
        '''
        if 'normalize' in method:
            self.data = self.normalize()
        elif 'feature_selection' in method:
            self.data = self.feature_selection()
        elif 'feature_engineering' in method:
            self.data = self.feature_engineering()
        elif method==[]:
            self.data = self.normalize()
            self.data = self.feature_selection()
            self.data = self.feature_engineering()
        else:
            print('no method')
        return self.data
        '''
        output type: pandas.DataFrame
        '''
class data_spilt():
    def __init__(self, data, label):
        '''
        data: data ,input tpye: pandas.DataFrame, pandas.Series
        label: target label, input type: pandas.DataFrame, pandas.Series
        model: model list, input type: list
        '''
        self.data = data
        self.label = label
    def train_test_split(self):
        '''
        split the data into train and test
        '''
        return self.x_train, self.x_test, self.y_train, self.y_test
        '''
        output type: pandas.DataFrame, numpy.array
        '''
    def train_val_split(self):
        '''
        split the data into train and validation
        '''
        return self.x_train, self.x_val, self.y_train, self.y_val
        '''
        output type: pandas.DataFrame, numpy.array
        '''