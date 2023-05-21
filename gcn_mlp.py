import numpy as np
import networkx as nx
import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn import Parameter
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_undirected

from models import *

dir = os.getcwd()
df_features = pd.read_csv(os.path.join (dir, 'dataset/elliptic_txs_features.csv'), header=None)
df_edge = pd.read_csv(os.path.join (dir, 'dataset/elliptic_txs_edgelist.csv'))
df_class = pd.read_csv(os.path.join (dir, 'dataset/elliptic_txs_classes.csv'))

# Setting Column name
print (df_features.head)
df_features.columns = ['id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
print (df_features.head)

nodes = list(set(df_edge['txId1']).union(set(df_edge['txId2'])).union(set(df_class['txId'])))
nodes_df = pd.DataFrame(nodes,columns=['id']).reset_index()
nodes_df.head


df_edges = df_edge.join(nodes_df.rename(columns={'id':'txId1'}).set_index('txId1'),on='txId1',how='inner').drop(columns=['index']) 
#        .join(nodes_df.rename(columns={'id':'txId2'}).set_index('txId2'),on='txId2',how='inner',rsuffix='2') \
#        .drop(columns=['txId1','txId2']) \
#        .rename(columns={'index':'txId1','index2':'txId2'})


df_edge_time = df_edges.join(df_features[['id','time step']].rename(columns={'id':'txId1'}).set_index('txId1'),on='txId1',how='left',rsuffix='1') \
.join(df_features[['id','time step']].rename(columns={'id':'txId2'}).set_index('txId2'),on='txId2',how='left',rsuffix='2')
df_edge_time['is_time_same'] = df_edge_time['time step'] == df_edge_time['time step2']
df_edge_time_fin = df_edge_time[['txId1','txId2','time step']].rename(columns={'txId1':'source','txId2':'target','time step':'time'})



node_label = df_class.rename(columns={'txId':'nid','class':'label'})[['nid','label']].sort_values(by='nid').merge(df_features[['id','time step']].rename(columns={'id':'nid','time step':'time'}),on='nid',how='left')
node_label['label'] =  node_label['label'].apply(lambda x: '3'  if x =='unknown' else x).astype(int)-1
node_label.info


merged_nodes_df = node_label.merge(df_features.rename(columns={'id':'nid','time step':'time'}).drop(columns=['time']),on='nid',how='left')
merged_nodes_df.info

nodes_df_tmp=merged_nodes_df[merged_nodes_df['time']==1].reset_index()
nodes_df_tmp['index']=nodes_df_tmp.index
df_edge_tmp = df_edge_time_fin.join(nodes_df_tmp.rename(columns={'nid':'source'})[['source','index']].set_index('source'),on='source',how='inner')\
    .join(nodes_df_tmp.rename(columns={'nid':'target'})[['target','index']].set_index('target'),on='target',how='inner',rsuffix='2') \
    .drop(columns=['source','target']) \
    .rename(columns={'index':'source','index2':'target'})

df_edge_tmp.head



# WORKING HEREEEEEEEE

all_nodes = list(set(df_edge['txId1']).union(set(df_edge['txId2'])).union(set(df_class['txId'])).union(set(df_features['id'])))
nodes_df = pd.DataFrame(all_nodes,columns=['id']).reset_index()
df_edge = df_edge.join(nodes_df.rename(columns={'id':'txId1'}).set_index('txId1'),on='txId1',how='inner') \
       .join(nodes_df.rename(columns={'id':'txId2'}).set_index('txId2'),on='txId2',how='inner',rsuffix='2') \
       .drop(columns=['txId1','txId2']) \
       .rename(columns={'index':'txId1','index2':'txId2'})


df_class = df_class.join(nodes_df.rename(columns={'id':'txId'}).set_index('txId'),on='txId',how='inner') \
        .drop(columns=['txId']).rename(columns={'index':'txId'})[['txId','class']]


df_features = df_features.join(nodes_df.set_index('id'),on='id',how='inner') \
        .drop(columns=['id']).rename(columns={'index':'id'})
df_features = df_features [ ['id']+list(df_features.drop(columns=['id']).columns) ]
df_edge_time = df_edge.join(df_features[['id','time step']].rename(columns={'id':'txId1'}).set_index('txId1'),on='txId1',how='left',rsuffix='1') \
.join(df_features[['id','time step']].rename(columns={'id':'txId2'}).set_index('txId2'),on='txId2',how='left',rsuffix='2')
df_edge_time['is_time_same'] = df_edge_time['time step'] == df_edge_time['time step2']
df_edge_time_fin = df_edge_time[['txId1','txId2','time step']].rename(columns={'txId1':'source','txId2':'target','time step':'time'})


node_label = df_class.rename(columns={'txId':'nid','class':'label'})[['nid','label']].sort_values(by='nid').merge(df_features[['id','time step']].rename(columns={'id':'nid','time step':'time'}),on='nid',how='left')
node_label['label'] =  node_label['label'].apply(lambda x: '3'  if x =='unknown' else x).astype(int)-1
node_label.head()


merged_nodes_df = node_label.merge(df_features.rename(columns={'id':'nid','time step':'time'}).drop(columns=['time']),on='nid',how='left')
merged_nodes_df.head()

print ('Preparing data')
train_dataset=[]
test_dataset=[]
local_feat = []
for i in range(49):
    nodes_df_tmp=merged_nodes_df[merged_nodes_df['time']==i+1].reset_index()
    nodes_df_tmp['index']=nodes_df_tmp.index
    df_edge_tmp = df_edge_time_fin.join(nodes_df_tmp.rename(columns={'nid':'source'})[['source','index']].set_index('source'),on='source',how='inner')\
        .join(nodes_df_tmp.rename(columns={'nid':'target'})[['target','index']].set_index('target'),on='target',how='inner',rsuffix='2') \
        .drop(columns=['source','target']) \
        .rename(columns={'index':'source','index2':'target'})
    x = torch.tensor(np.array(nodes_df_tmp.sort_values(by='index').drop(columns=['index','nid','label'])), dtype=torch.float)
    edge_index = torch.tensor(np.array(df_edge_tmp[['source','target']]).T, dtype=torch.long)
    edge_index = to_undirected(edge_index)
    mask = nodes_df_tmp['label']!=2
    
    agg_feat_idx = [f'agg_feat_{i}' for i in range(72)]
    nodes_local = nodes_df_tmp.drop(columns= agg_feat_idx)
    x_local = torch.tensor(np.array(nodes_local.sort_values(by='index').drop(columns=['index','nid','label'])), dtype=torch.float)

    
    y=torch.tensor(np.array(nodes_df_tmp['label']))

    if i+1<35:
        data = Data(x=[x, x_local],edge_index=edge_index, train_mask=mask, y=y)
        train_dataset.append(data)
        local_feat.append (x_local)
    else:
        data = Data(x= [x, x_local],edge_index=edge_index, test_mask=mask, y=y)
        test_dataset.append(data)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MGCN(num_node_features= data.num_node_features,hidden_channels=[50, 100])
# device = "cpu"
model.to(device)

     
patience = 50
lr = 0.01
epoches = 51
     
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7,0.3]).to(device))
criterion = torch.nn.NLLLoss(weight=torch.tensor([0.7,0.3]).to(device))


train_losses=[]
val_losses =[]
accuracies=[]
if1=[]
precisions=[]
recalls=[]
iterations=[]

max_if1 = 0
kappa_coef = 0
mathews_coef = 0
best_epoch = 0

print ("started training")
for epoch in range(epoches):
    
    model.train()
    train_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        _, pred = out[data.train_mask].max(dim=1)
        loss.backward()
        train_loss += loss.item() * data.num_graphs
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    

    if (epoch+1)%1==0:
        model.eval()
        ys, preds = [], []
        val_loss = 0
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out[data.test_mask], data.y[data.test_mask])
            val_loss += loss.item() * data.num_graphs
            _, pred = out[data.test_mask].max(dim=1)
            ys.append(data.y[data.test_mask].cpu())
            preds.append(pred.cpu())

        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
        val_loss /= len(test_loader.dataset)
        f1 = f1_score(y, pred, average=None)
        mf1 = f1_score(y, pred, average='micro')
        precision = precision_score(y, pred, average=None)
        recall = recall_score(y, pred, average=None)

        iterations.append(epoch+1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if1.append(f1[0])
        accuracies.append(mf1)
        precisions.append(precision[0])
        recalls.append(recall[0])

        if f1[0] > max_if1:
          best_epoch = epoch+1
          kappa_coef = cohen_kappa_score(y, pred)
          mathews_coef = matthews_corrcoef(y, pred)
          max_if1 = f1[0]

        print('Epoch: {:02d}, Train_Loss: {:.4f}, Val_Loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Illicit f1: {:.4f}, F1: {:.4f}'.format(epoch+1, train_loss, val_loss, precision[0], recall[0], f1[0], mf1))

print (f'Epoch for best if1: {best_epoch}')
print (f'max if1: {max_if1}')
print (f'kappa coefficient: {kappa_coef}')
print (f'Matthews correlation coefficient: {mathews_coef}')