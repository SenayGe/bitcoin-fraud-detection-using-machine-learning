from data_preprocessing import load_elliptic_dataset, split_train_test
from models import *
from utils import evaluate_performance, average_performance_per_timestep

# Loading dataset
X_df, y_df = load_elliptic_dataset() # Path to dataset is specified in 'data_preprocessing.py'

# Splitting dataset into training and test data (70/30)
X_train, X_test, y_train, y_test = split_train_test(X_df, y_df)


# Model fitting and Predictions

model_RF = supervised_model(X_train, y_train, model='RandomForest')
pred_RF = model_RF.predict(X_test)
f1_RF = evaluate_performance(y_test, pred_RF)
f1_RF_timestep = average_performance_per_timestep (X_test, y_test, pred_RF)
acc_RF = evaluate_performance(y_test, pred_RF, metric='accuracy')



model_XGB = supervised_model(X_train, y_train, model='XGBoost')
pred_XGB = model_RF.predict(X_test)
f1_XGB = evaluate_performance(y_test, pred_XGB)
f1_XGB_timestep = average_performance_per_timestep (X_test, y_test, pred_XGB)
acc_RF = evaluate_performance(y_test, pred_XGB, metric='accuracy')




model_LR = supervised_model(X_train, y_train, model='LogisticRegression')
pred_LR = model_RF.predict(X_test)
f1_LR = evaluate_performance(y_test, pred_LR)
f1_LR_timestep = average_performance_per_timestep (X_test, y_test, pred_LR)
acc_RF = evaluate_performance(y_test, pred_LR, metric='accuracy')