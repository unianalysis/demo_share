# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:33:49 2017

@author: wangronin
"""

import os
from mpi4py import MPI
from numpy import mean
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

pid = os.getpid()

comm = MPI.Comm.Get_parent()
comm_self = MPI.COMM_WORLD

model = comm.scatter(None, root=0)
data = comm.scatter(None, root=0)

index, training, test = data
test_set, target = test

# Model fitting in parallel...
model.fit(*training)

# Model predictions
test_pred = model.predict(test_set)
test_probs = model.predict_proba(test_set)

# Compute the performance measures
mse = mean((test_pred - target) ** 2.0)
acc = mean(test_pred == target)
_auc = roc_auc_score(target, test_probs[:, 1], 'weighted')
fpr, tpr, thresholds = roc_curve(target, test_probs[:, 1])

# Synchronization...
comm.Barrier()

# Gathering the fitted kriging model back
fitted = {
          'index': index, 
          'model': model, 
          'mse': mse, 
          'acc': acc, 
          'auc': _auc,
          'fpr': fpr,
          'tpr': tpr
          }
          
comm.gather(fitted, root=0)

comm.Disconnect()