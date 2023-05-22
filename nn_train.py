# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:57:04 2023

@author: User
"""

import structure as s
from structure import standardize, splitting_train_validation_StratifiedKFold, imblance_data, nn_model, convert_to_DataLoader, objective_nn, Training_nn, evaluation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score 
 
import optuna
import torch
import torch.optim as optim
import torch.nn as nn

# Difine parameter
random_state = 0
shuffle = False
input_size = None
n_epochs = 10
batch_size = 128
n_trials = 100

df = pd.read_csv('data.csv')
X = df.drop(['Bankrupt?'], axis=1)
y = df['Bankrupt?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
X_train, X_valid, y_train, y_valid = splitting_train_validation_StratifiedKFold(X_train, y_train, 5)
X_train, y_train = imblance_data(X_train, y_train, 3, random_state = random_state)


X_train, X_valid, X_test = X_train.values, X_valid.values, X_test.values
y_train, y_valid, y_test = y_train.values, y_valid.values, y_test.values

input_size = X_train.shape[1]

train_loader, valid_loader, test_loader = convert_to_DataLoader(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size)
ac, f1, pre, rec, auc, f_beta = Training_nn(train_loader, valid_loader, test_loader, input_size, n_epochs, batch_size, n_trials)

