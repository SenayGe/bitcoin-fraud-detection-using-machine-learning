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
    X = df_dataset.drop(columns=['id', 'class']) # to include node Id uncomment drop class only
    y = df_dataset['class']

    return X, y

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

    

def adj_mat_per_ts (X_train, ts_start, ts_end):
    """Retrun a list containitng the adjacency matrix for every timestep"""

    df_edges = pd.read_csv (data_edges)
    df_edges_labelled = df_edges[df_edges['TxId'].isin(X_train['id'])]
    adj_matrices = []
    for timestep in range(ts_start , ts_end + 1):
        num_tx = X_train[X_train['time_step'] == timestep].shape[0]
        adj_mat_ts = pd.DataFrame(np.zeros((num_tx, num_tx)))

        for i in range(num_tx):
            adj_mat_ts.loc[adj_mat_ts.index[i], df_edges_labelled.iloc[i]['txId2']] = 1

        adj_matrices.append(adj_mat_ts)

    return adj_matrices
