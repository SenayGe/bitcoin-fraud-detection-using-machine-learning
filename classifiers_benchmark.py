import os
from data_preprocessing import load_elliptic_dataset, split_train_test, adj_mat_per_ts
from models import *
from utils import evaluate_performance, average_performance_per_timestep, plot_performance_per_timestep
import warnings
warnings.filterwarnings('always')  

# Loading dataset
X_df, y_df, edges_df = load_elliptic_dataset() # Path to dataset is specified in 'data_preprocessing.py'

# Splitting dataset into training and test data (70/30)
X_train, X_test, y_train, y_test = split_train_test(X_df, y_df)

# test data with illicit nodes only
# X_test_illicit = X_test[y_test['class'] == 1]
y_test = y_test.to_frame()
y_test = y_test.reset_index(drop=True)
y_test_illicit = y_test[y_test['class'] == 1]
# index of illicit nodes
illicit_indices = y_test.index[y_test['class'] == 1].tolist()


# Model fitting and Predictions

model_RF = supervised_model(X_train, y_train, model='RandomForest')
pred_RF = model_RF.predict(X_test)
print (type(pred_RF))
# pred_RF_illicit = model_RF.predict(X_test_illicit)
pred_RF_illicit = list(map(lambda i: pred_RF[i], illicit_indices))
f1_RF = evaluate_performance(y_test, pred_RF)
f1_RF_timestep = average_performance_per_timestep (X_test, y_test, pred_RF)
f1_micro_RF = evaluate_performance(y_test, pred_RF, metric='f1_micro')
# acc_RF = evaluate_performance(y_test_illicit, pred_RF_illicit, metric='accuracy')
precision_RF = evaluate_performance(y_test, pred_RF, metric='precision')
recall_RF = evaluate_performance(y_test, pred_RF, metric='recall')

print ('------------------ Random Forest -----------------------')
print (f'Precision:  {precision_RF}')
print (f'Recall:  {recall_RF}')
print (f'F1_score:  {f1_RF}')
print (f'F1_Micro:  {f1_micro_RF}')

model_XGB = supervised_model(X_train, y_train, model='XGBoost')
pred_XGB = model_XGB.predict(X_test)
f1_XGB = evaluate_performance(y_test, pred_XGB)
f1_XGB_timestep = average_performance_per_timestep (X_test, y_test, pred_XGB)
f1_macro_XGB = evaluate_performance(y_test, pred_XGB, metric='f1_micro')
acc_XGB = evaluate_performance(y_test, pred_XGB, metric='accuracy')
precision_XGB = evaluate_performance(y_test, pred_XGB, metric='precision')
recall_XGB = evaluate_performance(y_test, pred_XGB, metric='recall')

print ('------------------ XGBoost model performance -----------------------')
print (f'Accuracy:  {acc_XGB}')
print (f'Precision:  {precision_XGB}')
print (f'Recall:  {recall_XGB}')
print (f'F1_score:  {f1_XGB}')
print (f'F1_Micro:  {f1_macro_XGB}')



model_LR = supervised_model(X_train, y_train, model='LogisticRegression')
pred_LR = model_LR.predict(X_test)
f1_LR = evaluate_performance(y_test, pred_LR)
f1_LR_timestep = average_performance_per_timestep (X_test, y_test, pred_LR)
f1_macro_LR = evaluate_performance(y_test, pred_LR, metric='f1_micro')
acc_LR = evaluate_performance(y_test, pred_LR, metric='accuracy')
precision_LR = evaluate_performance(y_test, pred_LR, metric='precision')
recall_LR = evaluate_performance(y_test, pred_LR, metric='recall')

print ('------------------ Losgistic Regression model performance -----------------------')
print (f'Accuracy:  {acc_LR}')
print (f'Precision:  {f1_LR}')
print (f'Recall:  {f1_LR}')
print (f'F1_score:  {f1_LR}')
print (f'F1_Micro:  {f1_macro_LR}')


print('lengthhhh', len(pred_LR))
print (X_test.shape)

model_f1_ts_dict = {'XGBoost': f1_XGB_timestep, 'Logistic Regression': f1_LR_timestep, 'Random Forest': f1_RF_timestep}


# Define colors
blue = '#0C2D48' #'#216597'
turquoise = '#011f4b' #'#5fc19e'
orange = '#eda84c'
red = '#e83622'

plot_performance_per_timestep(model_metric_dict=model_f1_ts_dict, last_train_time_step=34,
                              last_time_step=49, linewidth=3.5, figsize=(10, 5), labelsize=20, fontsize=22,
                              linestyle=['solid', "dotted", 'dashed'], linecolor=[turquoise, orange, red],
                              barcolor=blue, baralpha=0.3,
                              savefig_path=None)