import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np


def supervised_model (X_train, y_train, model):
    """ Use models from the sklearn library """

    if model == 'RandomForest':
        model_supervised = RandomForestClassifier()
    elif model == 'XGBoost':
        model_supervised = XGBClassifier()
    elif model == 'LogisticRegression':
        model_supervised = LogisticRegression()

    model_supervised.fit(X_train, y_train)
    return model_supervised

def unsupervised_model (X_train, y_train, model):
    """ Use models from the pyod library """
    model.fit(X_train, y_train)
    return model

def mlp ():
    pass