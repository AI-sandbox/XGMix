#2#

import sklearn.metrics
import numpy as np


def mad(preds,true):
    return np.mean(np.abs(preds-true))

func_maps = {"auroc":sklearn.metrics.accuracy_score,
             "roc":sklearn.metrics.accuracy_score,
             "mad":mad}

def return_metric_function(name):
    return func_maps[name]

