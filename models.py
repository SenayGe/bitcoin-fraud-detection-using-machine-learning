import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np


def supervised_model (model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def unsupervised_model ():
    pass

def neural_network ():
    pass