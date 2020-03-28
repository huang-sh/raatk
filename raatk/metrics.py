# -*- coding: utf-8 -*-

"""
metrics.py
~~~~~~~~~~~~
"""

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import LeaveOneOut



def _metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    result = {'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp, 
     			'precision': p, 'recall': r, "f1-score": f1,
      			'acc': acc, 'mcc': mcc, 'cm': cm}
    return result

def loo_metrics(clf, x, y):
    loo = LeaveOneOut()
    y_pred = cross_val_predict(clf, x, y, cv=loo)
    result = {0: _metrics(y, y_pred)}
    return result

def cv_metrics(clf, x, y, cv):
    cver = StratifiedKFold(n_splits=cv)
    y_pred = cross_val_predict(clf, x, y, cv=cver)
    result = {}
    for idx, (_, test_idx) in enumerate(cver.split(x, y)):
        sub_yt, sub_yp = y[test_idx], y_pred[test_idx]
        result[idx] = _metrics(sub_yt, sub_yp)
    return result
