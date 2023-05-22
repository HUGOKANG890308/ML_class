# example of stratified k-fold cross-validation with an imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from imblearn.over_sampling import SMOTE
import torch
# generate dataset
X,y=make_classification(n_samples=1000,n_classes=2,weights=[0.9,0.1],random_state=1)
print(X.shape,y.shape)
# define model
model=RandomForestClassifier(random_state=1)
# define evaluation procedure
cv=StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
# evaluate model
#tuple of arrays in the following order: (train,test)
scores=list()
smote=SMOTE()
for train_ix,test_ix in cv.split(X,y):
    # split dataset
    X_train,X_test=X[train_ix,:],X[test_ix,:]
    y_train,y_test=y[train_ix],y[test_ix]
    print('*'*10+'before smote'+10*'*')
    print(np.unique(y_train, return_counts=True))
    # oversample minority class
    X_train,y_train=smote.fit_resample(X_train,y_train)
    print(X_train.shape,y_train.shape)
    print('*'*10+'after smote'+10*'*')
    print(np.unique(y_train, return_counts=True))
    # fit model
    model.fit(X_train,y_train)
    # evaluate model
    yhat=model.predict(X_test)
    # store score
    f1_s=f1_score(y_test,yhat)
    scores.append(f1_s)
    print(f1_score(y_test,yhat,average=None))
    print(f1_score(y_test,yhat))
    print('f1_score=>',f1_s)
    print('acc=>',accuracy_score(y_test,yhat))
    print('---------------------------')
# report performance
print(f'f1=> mean: {np.mean(scores):.3f} ; Std:{np.std(scores):.3f}')
