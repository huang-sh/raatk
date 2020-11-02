# -*- coding: utf-8 -*-

"""
metrics.py
~~~~~~~~~~~~
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, plot_roc_curve



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


def cv_roc_curve_plot(clf, x, y, cv):
    tprs = []
    aucs = []        
    mean_fpr = np.linspace(0, 1, len(y))
    fig, ax = plt.subplots()
    cver = StratifiedKFold(n_splits=cv)
    for i, (train_idx, test_idx) in enumerate(cver.split(x, y)):
        clf.fit(x[train_idx], y[train_idx])
        viz = plot_roc_curve(clf, x[test_idx], y[test_idx], ax=ax, name=f"ROC fold {i}", alpha=.3, lw=1)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_fpr = np.linspace(0, 1, len(y))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=.8,
            label=r'Mean ROC (AUC = %0.4f)' % mean_auc)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ %0.4f std. dev.' % std_auc)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic")
    ax.legend(loc="lower right")
    name =  clf.__class__.__name__
    viz = RocCurveDisplay(fpr=mean_fpr, tpr=mean_tpr,
                            roc_auc=mean_auc, estimator_name=name)
    return viz


def loo_roc_curve_plot(clf, x, y):
    from functools import partial

    def loo_proba(i, x, y, clf):
        idx = list(range(len(y)))
        idx.pop(i)
        clf.fit(x[idx,:], y[idx])
        return clf.predict_proba(x[[i],:])[0, 1]
    func_ = partial(loo_proba, x=x, y=y, clf=clf)
    y_proba = [func_(i) for i in range(len(y))]
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    name =  clf.__class__.__name__
    ax = plt.figure().gca()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                        label='Chance', alpha=.8)
    viz = RocCurveDisplay(fpr=fpr, tpr=tpr, 
                        roc_auc=roc_auc, estimator_name=name)
    return viz.plot(name=name, ax=ax)


def roc_curve_plot(clf, x, y):
    y_prob = clf.predict_proba(x)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    name = clf.__class__.__name__
    viz = RocCurveDisplay(fpr=fpr, tpr=tpr, 
                        roc_auc=roc_auc, estimator_name=name)
    return viz.plot(name=name)
