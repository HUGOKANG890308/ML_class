from ML.train_test_spilt import train_test_selection
from ML.load_data import load_data
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from imblearn.metrics import sensitivity_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC
import os
import torch
import pandas as pd
class Metric():
    def __init__(self) :
        self.model=[]
        self.f1_1=[]
        self.f1_0=[]
        self.precision_0=[]
        self.precision_1=[]
        self.recall_0=[]
        self.recall_1=[]
        self.accuracy=[]
        self.model_name=[]
        self.TP=[]
        self.FP=[]
        self.FN=[]
        self.TN=[]
        self.seed=[]
        self.TP_rate=[]
        self.FN_rate=[]
        self.TN_rate=[]
        self.FP_rate=[]
        self.year=[]
        self.roc0=[]
        self.roc1=[]
    def save(self,save_path):
        df=pd.DataFrame({'model':self.model,'model_name':self.model_name,'f1_1':self.f1_1,'f1_0':self.f1_0,'precision_0':self.precision_0,'precision_1':self.precision_1,'recall_0':self.recall_0,'recall_1':self.recall_1,'accuracy':self.accuracy,'TP':self.TP,'FP':self.FP,'FN':self.FN,'TN':self.TN,'seed':self.seed,'TP_rate':self.TP_rate,'FN_rate':self.FN_rate,'TN_rate':self.TN_rate,'FP_rate':self.FP_rate,'year':self.year,'roc0':self.roc0,'roc1':self.roc1})
        df.to_csv(save_path,index=False)


class basic():
    def __init__(self,metric):
        self.metric=metric
        self.name = None  # add a default value for self.name
    def ML_model(self,ml_model):
        ml_model=ml_model.fit(self.x_train, self.y_train)
        self.metric.metric.model.append(ml_model)
        #若隨機抽取一個陽性樣本和一個陰性樣本，分類器正確判斷陽性樣本的值高於陰性樣本之機率 {\displaystyle =AUC}{\displaystyle =AUC}[1]。
        #numpy array [:,1]所有rows 取第一個cols
        try:
            self.metric.metric.roc0.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 0], average=None))
            self.metric.metric.roc1.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 1], average=None))
        except:
            self.metric.metric.roc0.append(0)
            self.metric.metric.roc1.append(0)
        self.y_pred= ml_model.predict(self.x_test)
        print('test_result')
        print(metrics.classification_report(self.y_test,self.y_pred))
        print('f1_score',f1_score(self.y_test, self.y_pred) )
        self.metric.metric.model_name.append(self.name)
        self.test_score(self.y_pred)
        print('random_state=',self.r)
        print('y_test.shape',self.y_test.shape)
        print('y_years=',self.years)
    def test_score(self,y_pred):
        print('y test',self.y_test.shape)
        if self.n_years==1:
            self.metric.metric.year.append(self.years+1)
        if self.n_years==2:
            period=str(self.years+1)+'~'+str(self.years+2) 
            self.metric.metric.year.append(period)
        if self.n_years==3:
            period=str(self.years+1)+'~'+str(self.years+3) 
            self.metric.metric.year.append(period)
        if self.n_years==4:
            period=str(self.years+1)+'~'+str(self.years+4) 
            self.metric.metric.year.append(period)    
        self.metric.metric.f1_0.append(f1_score(self.y_test,self.y_pred,average=None)[0])
        self.metric.metric.f1_1.append(f1_score(self.y_test,y_pred,average=None)[1])
        self.metric.metric.precision_0.append(precision_score(self.y_test, self.y_pred,average=None)[0])
        self.metric.metric.precision_1.append(precision_score(self.y_test,self.y_pred,average=None)[1])
        self.metric.metric.recall_0.append(sensitivity_score(self.y_test,self.y_pred,average=None)[0])
        self.metric.metric.recall_1.append(sensitivity_score(self.y_test,self.y_pred,average=None)[1])
        self.metric.metric.accuracy.append(accuracy_score(self.y_test, self.y_pred ))    
        self.metric.metric.TP.append(confusion_matrix(self.y_test,self.y_pred)[0][0])
        self.metric.metric.FP.append(confusion_matrix(self.y_test,self.y_pred)[1][0])
        self.metric.metric.TN.append(confusion_matrix(self.y_test,self.y_pred)[1][1])
        self.metric.metric.FN.append(confusion_matrix(self.y_test,self.y_pred)[0][1])
        self.metric.metric.TP_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][0]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        self.metric.metric.FN_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][1]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        self.metric.metric.TN_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][1]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        self.metric.metric.FP_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][0]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        self.metric.metric.seed.append(self.r)
class Logestic(basic):
    def __init__(self,dataset,seed,year,target,n_years,metric):
        super().__init__(metric)
        '''
        :dataset= data which you want, type=DataFrame
        :seed,random seed
        :year the last training year
        :target y target 
        
        :n_years how many years in test     
        '''

        self.target=target
        self.dataset=dataset
        self.r=seed
        self.n_years=n_years
        self.years=year
        #feature_selction(self.dataset, self.r,self.years,20,target=self.target)       
        train=train_test_selection(dataset=self.dataset,year=self.years,n_years=self.n_years,train=True)
        #train=train_test_selection(self.dataset,self.years,self.n_years,True)
        '''
        (dataset: Any, year: Any, n_years: Any, train: Any) -> None
        ;dataset= data which you want, type=DataFrame ;seed,random seed ;year the last training year ;n_years how many years in test
        ;if train=True return:trainingdata
        ;if train=False return:testingdata
        ;rtype: DataFrame
        '''
        training_data = train.__getitem__()
        testing_data=train_test_selection(self.dataset,self.years,self.n_years,False).__getitem__() 
        self.x_train,self.x_test=training_data.drop([self.target],axis=1),testing_data.drop([self.target],axis=1)
        self.y_train,self.y_test=training_data[self.target],testing_data[self.target]

    def logistic(self):
        log=LogisticRegression(random_state=self.r)
        self.name='logestic'
        
        print(self.name)
        super().ML_model(log)

class SVM(basic):
    def __init__(self,dataset,seed,year,target,n_years,metric):
       super().__init__(metric)
       '''
       :dataset= data which you want, type=DataFrame
       :seed,random seed
       :year the last training year
       :target y target 
       :n_years how many years in test     
       '''

       self.target=target
       self.dataset=dataset
       self.r=seed
       self.n_years=n_years
       self.years=year
       #feature_selction(self.dataset, self.r,self.years,20,target=self.target)       
       train=train_test_selection(dataset=self.dataset,year=self.years,n_years=self.n_years,train=True)
       #train=train_test_selection(self.dataset,self.years,self.n_years,True)
       '''
       (dataset: Any, year: Any, n_years: Any, train: Any) -> None
       ;dataset= data which you want, type=DataFrame ;seed,random seed ;year the last training year ;n_years how many years in test
        ;if train=True return:trainingdata
        ;if train=False return:testingdata
        ;rtype: DataFrame
       '''
       training_data = train.__getitem__()
       testing_data=train_test_selection(self.dataset,self.years,self.n_years,False).__getitem__() 
       self.x_train,self.x_test=training_data.drop([self.target],axis=1),testing_data.drop([self.target],axis=1)
       self.y_train,self.y_test=training_data[self.target],testing_data[self.target]
    def SVM(self):
        log=SVC(kernel='rbf',random_state =self.r)
        self.name='SVM'
        
        print(self.name)
        super().ML_model(log)

    
class Random_Forest(basic):
    def __init__(self,dataset,seed,year,target,n_years,metric):
       super().__init__(metric)
       '''
       :dataset= data which you want, type=DataFrame
       :seed,random seed
       :year the last training year
       :target y target 
       :n_years how many years in test     
       '''

       self.target=target
       self.dataset=dataset
       self.r=seed
       self.n_years=n_years
       self.years=year
       #feature_selction(self.dataset, self.r,self.years,20,target=self.target)       
       train=train_test_selection(dataset=self.dataset,year=self.years,n_years=self.n_years,train=True)
       #train=train_test_selection(self.dataset,self.years,self.n_years,True)
       '''
       (dataset: Any, year: Any, n_years: Any, train: Any) -> None
       ;dataset= data which you want, type=DataFrame ;seed,random seed ;year the last training year ;n_years how many years in test
        ;if train=True return:trainingdata
        ;if train=False return:testingdata
        ;rtype: DataFrame
       '''
       training_data = train.__getitem__()
       testing_data=train_test_selection(self.dataset,self.years,self.n_years,False).__getitem__() 
       self.x_train,self.x_test=training_data.drop([self.target],axis=1),testing_data.drop([self.target],axis=1)
       self.y_train,self.y_test=training_data[self.target],testing_data[self.target]
    def RF(self):
        log=RandomForestClassifier(random_state=self.r)
        self.name='RF'  
        print(self.name)
        super().ML_model(log)

class XGBOOST(basic):
    def __init__(self,dataset,seed,year,target,n_years,metric):
       super().__init__(metric)
       '''
       :dataset= data which you want, type=DataFrame
       :seed,random seed
       :year the last training year
       :target y target 
       :n_years how many years in test     
       '''

       self.target=target
       self.dataset=dataset
       self.r=seed
       self.n_years=n_years
       self.years=year
       #feature_selction(self.dataset, self.r,self.years,20,target=self.target)       
       train=train_test_selection(dataset=self.dataset,year=self.years,n_years=self.n_years,train=True)
       #train=train_test_selection(self.dataset,self.years,self.n_years,True)
       '''
       (dataset: Any, year: Any, n_years: Any, train: Any) -> None
       ;dataset= data which you want, type=DataFrame ;seed,random seed ;year the last training year ;n_years how many years in test
        ;if train=True return:trainingdata
        ;if train=False return:testingdata
        ;rtype: DataFrame
       '''
       training_data = train.__getitem__()
       testing_data=train_test_selection(self.dataset,self.years,self.n_years,False).__getitem__() 
       self.x_train,self.x_test=training_data.drop([self.target],axis=1),testing_data.drop([self.target],axis=1)
       self.y_train,self.y_test=training_data[self.target],testing_data[self.target]
    def XGB(self):
        log=XGBClassifier(random_state=self.r,
                                         tree_method='gpu_hist' if torch.cuda.is_available() else 'auto')
        self.name='XGB'  
        print(self.name)
        super().ML_model(log)
  
