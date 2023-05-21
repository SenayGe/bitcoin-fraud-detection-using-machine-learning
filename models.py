import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionmodels.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_undirected


'''Classical ML Algos'''

def supervised_model (X_train, y_train, model):
    """ Use models from the sklearn library """

    if model == 'RandomForest':         
        model_supervised = RandomForestClassifier()
    elif model == 'XGBoost':
        model_supervised = XGBClassifier()
    elif model == 'LogisticRegression':
        model_supervised = LogisticRegression()
    else:
        print('Invalid model name')
        return

    model_supervised.fit(X_train, y_train)
    return model_supervised

def unsupervised_model (X_train, y_train, model):
    """ Use models from the pyod library """
    model.fit(X_train, y_train)
    return model

def mlp ():
    pass


'''GCN + MLP'''
class MGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, use_skip=False):
        super(MGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0],hidden_channels[1])
        self.conv3 = GCNConv (hidden_channels[1], 150)
        self.linear1 = nn.Linear (94, 100) # We have 94 local features
        self.linear2 = nn.Linear(250, 100) # 110 is 100 from the linear + 10 from the gcn
        self.linear2_5 = nn.Linear(100, 81)
        self.linear3 = nn.Linear (81, 2)
        self.use_skip = use_skip
        if self.use_skip:
            self.weight = nn.init.xavier_normal_(Parameter(torch.Tensor(num_node_features, 2)))


    def forward(self, data):
        x_graph = data.x[0]
        x_linear = data.x[1]
        x = self.conv1(x_graph, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, data.edge_index)

        
        # Linearly transformed local feature data
        x_local = self.linear1 (x_linear)
        
        # Concatinating
        x_concat = torch.cat ((x, x_local), 1)
        
        # MLP
        x = x_concat.relu()
        x = self.linear2 (x)
        x = x.relu()
        x = self.linear2_5 (x)
        x = x.relu()
        x = self.linear3 (x)
    
#         x_concat = 
        
        if self.use_skip:
            x = F.softmax(x+torch.matmul(data.x, self.weight), dim=-1)
        else:
#             x = F.softmax(x, dim=-1)
            x = F.log_softmax(x, dim=1)
        return x

    def embed(self, data):
        x = self.conv1(data.x[0], data.edge_index)
        return x