import pandas 
import numpy as np
from sklearn.model_selection import train_test_split
class train_test_selection():
    def __init__(self,dataset,seed,train_size=0.8):
        '''        
        ;dataset= data which you want, type=DataFrame
        ;seed,random seed, type=int
        ;if train=True   return:trainingdata
        ;if train=False  return:testingdata
        ;rtype: DataFrame
        '''
        self.dataset=dataset
        self.seed=seed
        self.train_size=train_size
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.dataset,test_size=1-self.train_size,random_state=self.seed)
    
'''
def __getitem__(self):
           
        :param method: True,False
        :type method: booling
        :return: trainingdata, if train=True
                 testingdata, if train=False
        :rtype: DataFrame
        
        if self.train==True:
            return  self.trainingdata
        if self.train==False:
            return self.testingdata
'''