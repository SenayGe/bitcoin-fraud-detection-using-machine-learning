import os
import sys
import pandas as pd
import numpy as np

data_features = os.path.join (os.getcwd(), 'dataset/elliptic_txs_features.csv')
data_edges = os.path.join (os.getcwd(), 'dataset/elliptic_txs_edgelist.csv')
data_classes = os.path.join (os.getcwd(), 'dataset/elliptic_txs_classes.csv')

def merge_dataframes (df_features, df_classes, drop_unlabeled=True):
    df_merged = pd.merge (df_features, df_classes, left_on='id', right_on='txId', how='left')
    
    if drop_unlabeled:
        # print (df_merged.iloc([df_merged['class'] != 2]))
        df_merged = df_merged[df_merged['class'] != 2]
        df_merged.reset_index(drop=True)
    df_merged.drop(columns=['txId'], inplace=True)
    return df_merged
    
def load_elliptic_dataset (drop_unlabeled=True):
    df_features = pd.read_csv (data_features, header=None)
    df_classes = pd.read_csv (data_classes)
    df_edges = pd.read_csv (data_edges)
    
    # CLEANING DATASET

    # Reanming Class category (0: licit, 1: illicit, 2: unknown)
    df_classes.replace({'class': {'1': 1, '2': 0, 'unknown': 2}}, inplace=True)

    # Renaming feature names
    """
    features 0-93 represent local information while the remaining 
    72 features contain aggregated transaction infromation
    """
    df_features.columns = ['id', 'time_step'] + \
        [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
    
    # Combining the dataframes
    df_dataset = merge_dataframes (df_features, df_classes, drop_unlabeled)


    # Assigning X and y 
    # X = df_dataset.drop(columns=['id', 'class']) # to include node Id uncomment drop class only
    X = df_dataset.drop(columns=['class']) # to include node Id uncomment drop class only
    y = df_dataset['class']

    # Adding timestep column to df_edges
    df_edge_time = df_edges.join(df_features[['id','time_step']].rename(columns={'id':'txId1'}).set_index('txId1'),on='txId1',how='left',rsuffix='1') \
        .join(df_features[['id','time_step']].rename(columns={'id':'txId2'}).set_index('txId2'),on='txId2',how='left',rsuffix='2')
    df_edge_time = df_edge_time[['txId1','txId2','time_step']]

    return X, y, df_edge_time

def split_train_test (X, y):
    """ 
    Splitting the dataset to training and test data according to 70:30 ration.
    Temporally, this means the first 34 steps are part of the training data and the 
    rest part of the test data.
    """
    splitting_timestep = 34
    last_timestep = 49

    train_timesteps = list(range(splitting_timestep + 1))
    test_timesteps = list(range(splitting_timestep + 1, last_timestep + 1))

    train_idx = X[X['time_step'].isin(train_timesteps)].index
    test_idx = X[X['time_step'].isin(test_timesteps)].index

    # Splitting training and testing data
    X_train = X.loc[train_idx] #dataframe
    X_test = X.loc[test_idx] #dataframe

    y_train = y.loc[train_idx] #dataframe
    y_test = y.loc[test_idx] #dataframe

    return X_train, X_test, y_train, y_test 

    

def adj_mat_per_ts (X_train, df_edges, ts_start, ts_end):
    """Retrun a list containitng the adjacency matrix for every timestep"""


    # labelled_idx = df_edges[df_edges['txId1'].isin(X_train['id'])].index
    # df_edges_labelled = df_edges.loc[labelled_idx]
    df_edges_labelled = df_edges[df_edges['txId1'].isin(X_train['id'])]

 
    num_tx = X_train.shape[0]
    total_txs = list(X_train['id'])

    adj_matrices = []

    for timestep in range(ts_start , ts_end+1):
        X_train_ts = X_train[X_train['time_step'] == timestep]
       
        txs = list(X_train_ts['id']) # txs in this timestep

        # df_edges_labelled_ts = df_edges_labelled[df_edges_labelled['time_step'] == timestep]
        df_edges_labelled_ts = df_edges_labelled[df_edges_labelled['time_step'] == timestep]

        print('df_edges_labelled_ts:  ', df_edges_labelled_ts.shape[0] )
        print('X shape:  ', len(txs))
       

        
        # adj_mat = pd.DataFrame(np.zeros((num_tx, num_tx)), index = total_txs, columns =total_txs)
        
        # for index, row in df_edges_labelled_ts.iterrows():
        #     adj_mat.loc[ row['txId1'], row['txId2']] = 1
        # adj_mat_ts = adj_mat.loc[txs, txs]


        # using crosstab 
     
        adj_mat_ts = pd.crosstab(df_edges_labelled_ts.txId1, df_edges_labelled_ts.txId2)
        adj_mat_ts = adj_mat_ts.reindex(index = txs, columns=txs, fill_value=0)
        
        # idx = adj_mat.columns.union(adj_mat.index)
      

        # adj_mat_ts = adj_mat_ts.loc[ adj_mat_ts.index.isin(txs), :]
        
        adj_matrices.append(adj_mat_ts)

    return adj_matrices


list1 = [5, 6, 4]
list2 = [1, 3 , 8]

df = pd.DataFrame(list(zip(list1, list2)), columns =['txId1', 'txId2'])

adj_mat_ts = pd.crosstab(df.txId1, df.txId2)
adj_mat_ts = adj_mat_ts.reindex(index = [5, 1, 6, 3, 4, 8], columns= [5, 1, 6, 3, 4, 8], fill_value=0)

print (adj_mat_ts)