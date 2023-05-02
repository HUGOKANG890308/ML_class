from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import torch
from sklearn.model_selection import train_test_split
# generate dataset
X,y=make_classification(n_samples=1000,n_classes=2,weights=[0.9,0.1],random_state=1)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
def basic_ml(using_model={'xgb':XGBClassifier(),'rf':RandomForestClassifier()}
             ,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test):
    '''
    using_model: input using model ; type: dictionary
    x_train: input x_train ; type: numpy.ndarray or dataframe
    y_train: input y_train ; type: numpy.ndarray or dataframe
    x_test: input x_test ; type: numpy.ndarray  or dataframe
    y_test: input y_test ; type: numpy.ndarray  or dataframe
    '''
    for i in using_model:
        model=using_model[i]
        model.fit(x_train,y_train)
        yhat=model.predict(x_test)
        print(f'using {i} model')
        print(f'f1_score=>{f1_score(y_test,yhat)}')
        #厚儀的evaluation metric
        print('---------------------------')

basic_ml(using_model={'xgb':XGBClassifier(),'rf':RandomForestClassifier()},x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
