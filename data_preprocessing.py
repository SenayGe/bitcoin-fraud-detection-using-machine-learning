import os
import sys
import pandas as pd

data_features = os.path.join (os.getcwd(), 'dataset/elliptic_txs_features.csv')
data_edges = os.path.join (os.getcwd(), 'dataset/elliptic_txs_edgelist.csv')
data_classes = os.path.join (os.getcwd(), 'dataset/elliptic_txs_classes.csv')

def merge_dataframes (df_features, df_classes, drop_unlabeled=True):
    df_dataset = pd.merge_dataset (df_features, df_classes, left_on='id', right_on='txtId', how='id')
    df_dataset.drop(columns=['txtId'], inplace=True)
    if drop_unlabeled:
        df_combined = df_combined[df_combined['class'] != 2].reset_index(drop=True)

    return df_dataset
    
def load_elliptic_dataset (drop_unlabeled=True):
    df_features = pd.read_csv (data_features)
    df_edges = pd.read_csv (data_edges)
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
    df_dataset = merge_dataframes (df_classes, df_features, drop_unlabeled)


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
    X_train = X.loc(train_idx) #dataframe
    X_test = X.loc(test_idx) #dataframe

    y_train = y.loc(train_idx) #dataframe
    y_test = y.loc(test_idx) #dataframe

    return X_train, X_test, y_train, y_test 

    



