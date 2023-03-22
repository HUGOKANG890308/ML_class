import numpy as np
import pandas as pd
from sklearn.preprocessing  import StandardScaler
from sklearn.preprocessing  import MinMaxScaler
import os
class load_data():
    def __init__(self, path):
        '''
        path: the path of the data
        input_type:DataFrame
        output_type:DataFrame
        '''
        self.path = path
        self.data = pd.read_csv(self.path)
        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop=True)
    def scaler(self,scaler_type):
        '''
        scaler_type: "None","standardscaler","minmaxscaler"
        input_type:DataFrame
        output_type:DataFrame
        '''
        self.data_scaler=self.data.copy()
        if scaler_type == "None":
            pass
        elif scaler_type == "standardscaler":
            for i in range(3,len(self.data.columns)):
                self.data_scaler.iloc[:,i]=StandardScaler().fit_transform(self.data.iloc[:,i].values.reshape(-1,1))            
        elif scaler_type == "minmaxscaler":
            for i in range(3,len(self.data.columns)):
                self.data_scaler.iloc[:,i]=MinMaxScaler().fit_transform(self.data.iloc[:,i].values.reshape(-1,1))
        else:
            print("Please input correct scaler type")
        return self.data_scaler
    def get_data(self):
        return self.data
    
