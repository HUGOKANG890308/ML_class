from ML.train_test_spilt import train_test_selection
from ML.load_data import load_data
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from imblearn.metrics import sensitivity_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC
import os
os.chdir("C:\\Users\\User\\Desktop\\ML_class\\ML")
import torch
import pandas as pd
path='weatherAUS.csv'
data=load_data(path)
data=data.get_data()
for i in(data.columns[:-1]):
    if data[i].dtype=="object":
        data=data.drop(i,axis=1)
traingdata,testdata=train_test_selection(data,seed=0)
traingdata,testdata=train_test_selection(data,seed=1,train_size=0.8)
x_train=traingdata.iloc[:,3:]
y_train=traingdata.iloc[:,2]
x_test=testdata.iloc[:,3:]
y_test=testdata.iloc[:,2]