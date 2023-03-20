import os
os.chdir("C:\\Users\\User\\Desktop\\ML_class")
from ML.data_spilt import train_test_selection
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
import torch
import pandas as pd
import numpy as np
path='weatherAUS.csv'
data=load_data(path)
data=data.get_data()
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})
for i in(data.columns):
    if data[i].dtype=="object":
        data=data.drop(i,axis=1)
x_train, x_test, y_train, y_test=train_test_selection(dataset=data,target='RainTomorrow',seed=0,train_size=0.8).get_data()
Classifier=XGBClassifier(random_state=0, tree_method='gpu_hist' if torch.cuda.is_available() else 'auto')
Classifier.fit(x_train,y_train)
y_pred=Classifier.predict(x_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred))
print("AUC:",metrics.roc_auc_score(y_test, y_pred))
print("Sensitivity:",sensitivity_score(y_test, y_pred))
print("Specificity:",metrics.recall_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print("F1:",f1_score(y_test, y_pred, average='binary', pos_label=1, labels=np.unique(y_pred)))
print("F1:",f1_score(y_test, y_pred, average='binary', pos_label=0, labels=np.unique(y_pred)))

