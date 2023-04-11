import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split

import xgboost as xgb


#Preprocessing

# load data
df = pd.read_csv('data.csv (1).zip')
df.head()

#Define preprocessor
# Fill sequencial data
def fillMedian(data):
    data = data.select_dtypes(exclude=['object'])
    cols = data.columns

    for col in cols:
        data[col].fillna(value=df[col].median(), inplace=True)
    return data

# Fill catogory data
def fillcat(data):
    data = data.select_dtypes(exclude=['float', 'int'])

    cols = data.columns
    one_hot_cols = []
    ord_cols = []

    for col in cols:
        data[col].fillna(value=data[col].mode, inplace=True)
        if len(data[col].unique()) < 10:
            one_hot_cols.append(col)
        else:
            ord_cols.append(col)
    
    label_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value', 
        unknown_value=-1
        )
    one_hot_transformer = OrdinalEncoder(
        handle_unknown='ignore'
    )

    preprocessor =  ColumnTransformer(
        transformers=[
        ('num', label_transformer, ord_cols),
        ('cat', one_hot_transformer, one_hot_cols)
        ]
    )
   
    _data = preprocessor.fit_transform(data)
    return _data

X= df.drop(['Bankrupt?'], axis=1)
y= df['Bankrupt?']

num_transformer = FunctionTransformer(fillMedian)
cat_transsformer = FunctionTransformer(fillcat)
columns = X.columns

preprocessor = ColumnTransformer(
    transformers=[
    ('num', num_transformer, columns),
    ('cat', cat_transsformer, columns),
    ]
)
X = preprocessor.fit_transform(X)

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, t_test = train_test_split(X, y, test_size=0.2)





# Build XGBoost model
class XGB():

    def __init__(self):
        self.model = xgb.XGBClassifier()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

# Define main metric


model = XGB() 
model.fit(X_train, y_train)
pred = model.predict(X_test)  
