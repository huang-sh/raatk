# -*- coding: utf-8 -*-
"""   
    :Author: huangsh
    :Date: 19-1-15 上午11:07
    :Description:

"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, LeaveOneOut
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, roc_curve


class Evaluate:
    def __init__(self, model, x, y, probability=False):
        self.probability = probability
        self.model = model
        self.x = x
        self.y = y
        
    def loo(self):
        lo = LeaveOneOut()
        clf = self.model
        X, y = self.x, self.y        
        ss = lo.split(X)
        y_pre_arr = np.zeros(len(y))
        y_prob_arr = np.zeros(len(y))
        for train_idx, test_idx in ss:
            x_train, y_train = X[train_idx], y[train_idx]
            x_test, y_test = X[test_idx], y[test_idx]
            fit_clf = clf.fit(x_train, y_train)
            y_true, y_pre = y_test, fit_clf.predict(x_test)
            y_pre_arr[test_idx] = y_pre
            if self.probability:
                y_prob_arr[test_idx] = clf.predict_proba(x_test)[:,1]
        self.sub_metric = [self.metrics_(y, y_pre_arr)]
        self.mcm = [multilabel_confusion_matrix(y, y_pre_arr)]
        self.cv_eval = None
        self.y_true = [y]
        self.y_pre = [y_pre_arr]
        self.y_prob = [y_prob_arr]
    
    def kfold(self, k):
        skf = StratifiedKFold(n_splits=k,)
        ss = skf.split(self.x, self.y)
        clf = self.model
        X, y = self.x, self.y
        metric_ls = []
        mcm = []
        y_true_ls = []
        y_pre_ls = []
        y_prob_ls = []
        for train_idx, test_idx in ss:
            x_train, y_train = X[train_idx], y[train_idx]
            x_test, y_test = X[test_idx], y[test_idx]
            fit_clf = clf.fit(x_train, y_train)
            y_true, y_pre = y_test, fit_clf.predict(x_test)
            sub_cm = multilabel_confusion_matrix(y_true, y_pre)
            mcm.append(sub_cm)
            y_true_ls.append(y_true)
            y_pre_ls.append(y_pre)
            sub_metric = self.metrics_(y_true, y_pre) # sn, sp, presision, acc, mcc, fpr, tpr,
            metric_ls.append(sub_metric)
            if self.probability:
                y_prob = fit_clf.predict_proba(x_test)[:, 1]
                y_prob_ls.append(y_prob)
             ## self.mcm=[]; self.mcm.append(sub_cm)为啥不行？？？ 
        # self.metric = np.mean(metric_ls, axis=0)
        self.mcm = mcm
        self.sub_metric = metric_ls
        self.y_true = y_true_ls
        self.y_pre = y_pre_ls
        self.y_prob = y_prob_ls

    def holdout(self, test_size):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size)
        fit_clf = self.model.fit(x_train, y_train)
        y_true, y_pre = y_test, fit_clf.predict(x_test)
        self.sub_metric = [self.metrics_(y_true, y_pre)]
        self.mcm = [multilabel_confusion_matrix(y_true, y_pre)]
        # self.fpr_tpr_auc_cm(y, y_pre)

    def metrics_(self, y_true, y_pre):
        le = LabelEncoder()
        y_true, y_pre = y_true.ravel(), y_pre.ravel()
        unique_label = np.unique(y_true)
        le.fit(unique_label)
        idx_label = le.transform(unique_label)
        y_true = le.transform(y_true)
        y_pre = le.transform(y_pre)
        mcm = multilabel_confusion_matrix(y_true, y_pre, labels=idx_label)
        tn = mcm[:, 0, 0]
        tp = mcm[:, 1, 1]
        fn = mcm[:, 1, 0]
        fp = mcm[:, 0, 1]
        sn = tp / (tp + fn) # recall / sensitivity
        sp = tn / (tn + fp) # specificity
        ppv = tp / (tp + fp) # presision
        acc = (tp + tn) / (tp + tn + fp + fn)
        mcc = (tp * tn -fp * fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return [acc, sn, sp, ppv, mcc]

    def get_eval_idx(self):
        # acc, sn, sp, ppv, mcc = self.metric 
        oa = sum([sum(i[:, 1, 1]) for i in self.mcm]) / sum([len(i) for i in self.y_true])
        metric_dic = {'y_true': self.y_true, "y_pre": self.y_pre, 'y_prob': self.y_prob,
                     "mcm":self.mcm, "sub_metric": self.sub_metric, 'OA': oa}
        return metric_dic

    def fpr_tpr_auc_cm(self, y, y_pre):
        self.fpr, self.tpr, self.auc, self.mcm = [], [], [], []
        self.y, self.y_pre = [], []
        fpr, tpr, _ = roc_curve(y, y_pre)
        self.fpr.append(fpr)
        self.tpr.append(tpr)
        self.auc.append(auc(fpr, tpr))
        mcm = multilabel_confusion_matrix(y, y_pre)
        self.mcm.append(mcm)
        self.y.append(y)
        self.y_pre.append(y_pre)

class SvmClassifier:

    def __init__(self, kernel='rbf', C=1, gamma=0.1, cv=5):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.clf = SVC(class_weight='balanced', probability=True,
                        C=1.0, kernel='rbf', gamma='scale',)  # cache_size=500

    def train(self, x_train, y_train):
        svm = self.clf.set_params(C=self.C, gamma=self.gamma)
        svm = self.clf.fit(x_train, y_train)
        return svm

class KnnClassifier:

    def __init__(self, cv=5, n_neighbors=6, ):
        self.n_neighbors = n_neighbors
        self.cv = cv

    def train(self, x_train, y_train):
        clf = KNeighborsClassifier(self.n_neighbors, weights='distance', n_jobs=-1)
        clf = clf.fit(x_train, y_train)
        return clf

class RfClassifier:

    def __init__(self, cv=5, n_estimators=30):
        self.n_neighbors = n_estimators
        self.cv = cv

    def train(self, x_train, y_train):
        clf = RandomForestClassifier(n_estimators=30, class_weight='balanced', n_jobs=-1)
        clf = clf.fit(x_train, y_train)
        return clf
