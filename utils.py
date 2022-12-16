from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from data_preprocessing import load_elliptic_dataset
from matplotlib import pyplot as plt
import numpy as np
import os

def calc_performance_score (y_true, y_pred, metric='f1'):
    performance_scores = {'accuracy': accuracy_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred, pos_label=1),
                   'f1_micro': f1_score(y_true, y_pred, average='micro'),
                   'f1_macro': f1_score(y_true, y_pred, average='macro'),
                   'precision': precision_score(y_true, y_pred, pos_label=1), 'recall': recall_score(y_true, y_pred),
                   'roc_auc': roc_auc_score(y_true, y_pred)}
    
    # Return a specific metric
    model_score = performance_scores[metric]

    return model_score

def evaluate_performance(y_true, y_pred, metric='f1'):

    # if not isinstance(y_pred,list):
    #     return calc_performance_score(y_true.astype('int'), y_pred, metric)

    # model_scores = []
    # for y_i in y_pred:
    #     score = calc_performance_score(y_true.astype('int'), y_i, metric)
    #     model_scores.append(score)

    # avg_score = np.mean(model_scores)

    return calc_performance_score(y_true.astype('int'), y_pred, metric)


def average_performance_per_timestep(X_test, y_test, y_pred, metric= 'f1'):
    """ Calculates the average performance score per timestep"""

    first_test_timestep = min(X_test['time_step']) # Timestep where test data starts
    last_time_step = max(X_test['time_step'])

    all_scores = []
    # for y_pred in y_preds:
    score_ts = [] # Score in each timestep
    for time_step in range(first_test_timestep , last_time_step + 1):
        time_step_idx = np.flatnonzero(X_test['time_step'] == time_step)
        y_true_ts = y_test.iloc[time_step_idx]
        y_pred_ts = [y_pred[i] for i in time_step_idx]
        score_ts.append(calc_performance_score(y_true_ts.astype('int'), y_pred_ts, metric))
    all_scores.append(score_ts)

    avg_f1 = np.array([np.mean([f1_scores[i] for f1_scores in all_scores]) for i in range(15)])

    return avg_f1 

def calc_occurences_per_timestep():
    X, y, _ = load_elliptic_dataset()
    X['class'] = y
    occ = X.groupby(['time_step', 'class']).size().to_frame(name='occurences').reset_index()
    return occ


def plot_performance_per_timestep(model_metric_dict, last_train_time_step=34,last_time_step=49, model_std_dict=None, fontsize=23, labelsize=18, figsize=(20, 10),
                                  linestyle=['solid', "dotted", 'dashed'], linecolor=["green", "orange", "red"],
                                  barcolor='lightgrey', baralpha=0.3, linewidth=1.5, savefig_path=None):
    occ = calc_occurences_per_timestep()
    illicit_per_timestep = occ[(occ['class'] == 1) & (occ['time_step'] > 34)]

    timesteps = illicit_per_timestep['time_step'].unique()
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    i = 0
    for key, values in model_metric_dict.items():
        if key != "XGBoost":
            key = key.lower()
        ax1.plot(timesteps, values, label=key, linestyle=linestyle[i], color=linecolor[i], linewidth=linewidth)
        if model_std_dict != None:
            ax1.fill_between(timesteps, values + model_std_dict[key], values - model_std_dict[key],
                             facecolor='lightgrey', alpha=0.5)
        i += 1

    ax2.bar(timesteps, illicit_per_timestep['occurences'], color=barcolor, alpha=baralpha, label='\# illicit')
    ax2.get_yaxis().set_visible(True)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2.grid(False)

    ax1.set_xlabel('Time step', fontsize=fontsize)
    ax1.set_ylabel('Illicit F1', fontsize=fontsize)
    ax1.set_xticks(range(last_train_time_step+1,last_time_step+1))
    ax1.set_yticks([0,0.25,0.5,0.75,1])
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, fontsize=fontsize, facecolor="#EEEEEE")

    ax1.tick_params(direction='in')

    ax2.set_ylabel('Num. samples', fontsize=fontsize)

    if savefig_path == None:
        plt.show()
    else:
        plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0)
