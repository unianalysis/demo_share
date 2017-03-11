# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
"""

from __future__ import division   # important! for the float division

import pdb
import time
import sys
import os

import numpy as np
from numpy import array, tile, c_, mean, zeros

from scipy import interp

from rpy2.robjects import r
from rpy2.robjects import pandas2ri

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import auc

from pandas import DataFrame

import cPickle as cp

pandas2ri.activate()

# plot settings
plt.ioff()
fig_width = 22
fig_height = 22 * 9 / 16
_color = ['r', 'm', 'b', 'g', 'c']

rcParams['font.size'] = 15

# setup working directory: change it to the absolute path on your machine
os.chdir(os.path.expanduser('~')  + '/Dropbox/昊唯分析/Demo')

# ----------------------- Loading the data set from R ------------------------
r['load']('./data/model_training.RData')

df = pandas2ri.ri2py(r['data'])

X = df.iloc[:, 0:-1].as_matrix()
y = array(df['is_error'], dtype='int')

n_sample, n_feature = X.shape
n_error = sum(y)

print 'total records: {}'.format(n_sample)
print 'no defect records: {}'.format(n_error)
print 'defect records: {}'.format(n_sample - n_error)


# -------------------------------- Modellling ---------------------------------
# setup saving paths
data_path = './data'
fig_path = './figure/modelling'

try:
    os.makedirs(fig_path)
except:
    pass
 
model_list = ['naive bayes', 'logit', 'random forest', 'svm']

n_folds = 5
n_rep = 10
n_model = len(model_list)

# evaluation measures
auc_score = zeros((n_model, n_folds, n_rep))
acc_rate = zeros((n_model, n_folds, n_rep))
mse_score = zeros((n_model, n_folds, n_rep))
rate = {}

# Try to import MPI module
try:
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_process = comm.Get_size()
    
except:
    raise Exception('No module named mpi4py!')

# perform the mode cross-validation
for i, model in enumerate(model_list): 
    
    for j in range(n_rep):
        
        print 'model: {}, Repetition: {}'.format(model, j+1)
        
        skf = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
        index = [(train_index, test_index) for train_index, test_index in skf]
        
        training_set = [(X[train_index, :], y[train_index]) \
            for train_index, test_index in index]
        test_set = [(X[test_index, :], y[test_index]) \
            for train_index, test_index in index]
        
        t_start = time.time()
        if model == 'svm':
            
            # first K fold cross validation to evaluate the model 
            kernel = ['linear', 'rbf', 'poly', 'sigmoid']
            kernel = 'rbf'
            n_kernel = len(kernel)
        
            C = 1.0             # SVM regularization parameter
            gamma = 0.7         # RBF parameter
            degree = 3          # degree of the polynomial
            models = [svm.SVC(kernel=kernel, C=C, degree=degree, 
                              gamma=gamma, probability=True) \
                              for k in range(n_folds)]
                    
        elif model == 'naive bayes':
            models = [naive_bayes.BernoulliNB() for k in range(n_folds)]
        
        elif model == 'random forest':
            models = [RandomForestClassifier(n_estimators=500, 
                                             class_weight='auto') \
                                             for k in range(n_folds)]
                    
        elif model == 'logit':
            models = [LogisticRegression(class_weight='auto', max_iter=200, 
                                         solver='lbfgs') \
                                         for k in range(n_folds)]
        
        elif model == 'LDA':
            pass
        
        # Spawning processes to test kriging mixture
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['fitting.py'], 
                                   maxprocs=n_folds)
    
        # scatter the models and data
        comm.scatter(models, root=MPI.ROOT)
        comm.scatter([(k, training_set[k], test_set[k]) \
            for k in range(n_folds)], root=MPI.ROOT)
        
        # Synchronization while the children process are performing 
        # heavy computations...
        comm.Barrier()
            
        # Gether the fitted model from the childrenn process
        # Note that 'None' is only valid in master-slave working mode
        results = comm.gather(None, root=MPI.ROOT)
            
        # free all slave processes
        comm.Disconnect()
        
        # tic toc
        t_stop = time.time()
        print 'takes {0:.3f}mins'.format((t_stop - t_start)/60)
        
        # register the measures
        scores = DataFrame([[d['index'], d['acc'], d['mse'], d['auc']] \
            for d in results], columns=['index', 'acc', 'mse', 'auc'])
        scores.sort('index', inplace=True)
        
        acc_rate[i, :, j] = scores['acc']
        auc_score[i, :, j] = scores['auc']
        mse_score[i, :, j] = scores['mse']
        
        # record the fpr, tpr rates
        rate[j] = {d['index']: c_[d['fpr'], d['tpr']] for d in results}
    
    print 'accuracy: {}'.format(mean(acc_rate[i, :, :]))
    print 'auc score: {}'.format(mean(auc_score[i, :, :]))
    print 'mse: {}'.format(mean(mse_score[i, :, :]))
    
    fig, axes = plt.subplots(2, 5, figsize=(fig_width, fig_height), dpi=100)
    axes.shape = (n_rep, )
    
    for k, ax in enumerate(axes):
        ax.grid(True)
        ax.hold(True)
        
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for n in range(n_folds):
            fpr = rate[k][n][:, 0]
            tpr = rate[k][n][:, 1]
            roc_auc = auc_score[i, n, j]
            
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            
            ax.plot(fpr, tpr, lw=1, color = _color[n],
                    label='ROC fold %d (area = %0.2f)' % (n+1, roc_auc))
                    
        mean_tpr /= n_folds
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
                    
        ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='random gusess')
        ax.plot(mean_fpr, mean_tpr, 'k--', 
                label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_title('Rep {}'.format(k+1))
        ax.legend(loc="lower right", prop={'size': 10})
    
    fig.text(0.5, 0.04, 'False Positive Rate', ha='center', va='center', 
             fontsize=17)
    fig.text(0.1, 0.5, 'True Positive Rate', ha='center', 
             va='center', rotation='vertical', fontsize=17)
    fig.suptitle("ROC of model {}".format(model))
    fig.savefig(fig_path + '/ROC-{}.pdf'.format(model))
    
# Save the data
f = file(data_path + '/cv_comparison.dat', 'w')
cp.dump({'model_list': model_list,
         'n_folds': n_folds,
         'n_reptition': n_rep,
         'acc_rate': acc_rate,
         'mse_rate': mse_score,
         'auc_score': auc_score}, f)


# data plotting
fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))

acc_data = [acc_rate[i, :, :].flatten() for i in range(n_model)]
auc_data = [auc_score[i, :, :].flatten() for i in range(n_model)]
mse_data = [mse_score[i, :, :].flatten() for i in range(n_model)]

axes[0].boxplot(acc_data)
axes[1].boxplot(mse_data)
axes[2].boxplot(auc_data)

axes[0].set_title('accuracy')
axes[1].set_title('MSE')
axes[2].set_title('AUC')

for ax in axes:
    ax.grid(True)
    ax.hold(True)
    ax.set_xticklabels(model_list)
    
fig.savefig(fig_path + '/cv-comparison.pdf')
    
pdb.set_trace()
