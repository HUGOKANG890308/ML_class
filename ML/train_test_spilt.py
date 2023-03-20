import pandas 
import numpy as np
class train_test_selection():
    def __init__(self,dataset,year,n_years,train):
        '''        
        ;dataset= data which you want, type=DataFrame
        ;seed,random seed
        ;year  the last training year
        ;n_years how many years in test     
        ;if train=True   return:trainingdata
        ;if train=False  return:testingdata
        ;rtype: DataFrame
        '''
    def __getitem__(self):
        '''        
        :param method: True,False
        :type method: booling
        :return: trainingdata, if train=True
                 testingdata, if train=False
        :rtype: DataFrame
        '''
        if self.train==True:
            return  self.trainingdata
        if self.train==False:
            return self.testingdata

