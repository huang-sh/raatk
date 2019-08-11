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


class Evaluate:
    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y

    def loo(self):
        lo = LeaveOneOut()
        clf = self.model
        X, y = self.x, self.y        
        ss = lo.split(X)
        y_pre_arr = np.zeros(len(y))
        for train_idx, test_idx in ss:
            x_train, y_train = X[train_idx], y[train_idx]
            x_test, y_test = X[test_idx], y[test_idx]
            fit_clf = clf.fit(x_train, y_train)
            y_true, y_pre = y_test, fit_clf.predict(x_test)
            y_pre_arr[test_idx] = y_pre
        metric = self.metrics_(y, y_pre_arr)
        cm = multilabel_confusion_matrix(y, y_pre_arr)
        return metric, cm
    
    def kfold(self, k):
        skf = StratifiedKFold(n_splits=k, random_state=1)
        ss = skf.split(self.x, self.y)
        clf = self.model
        all_metrics = 0
        X, y = self.x, self.y
        for train_idx, test_idx in ss:
            x_train, y_train = X[train_idx], y[train_idx]
            x_test, y_test = X[test_idx], y[test_idx]
            fit_clf = clf.fit(x_train, y_train)
            y_true, y_pre = y_test, fit_clf.predict(x_test)
            metric = self.metrics_(y_true, y_pre) # sn, sp, presision, acc, mcc, fpr, tpr,
            all_metrics = np.add(all_metrics, metric)
        k_mean_metric = all_metrics / k
        return k_mean_metric

    def holdout(self, test_size):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, shuffle=True,
                                                random_state=1, test_size=test_size)
        fit_clf = self.model.fit(x_train, y_train)
        y_true, y_pre = y_test, fit_clf.predict(x_test)
        metric = self.metrics_(y_true, y_pre)
        cm = multilabel_confusion_matrix(y_true, y_pre)
        return metric, cm

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
        mcc = (tp * tn -fp * fn) / np.sqrt((tp+fp)*(tp+fn)*(tn*fp)*(tn+fn))
        return acc, sn, sp, ppv, mcc
        

class SvmClassifier:

    def __init__(self, param_grid=None, kernel='rbf', C=1, gamma=0.1, cv=5,
                 grid_search=True, ):
        self.cv = cv
        self.param_grid = param_grid if param_grid else {}
        self.is_grid_search = grid_search
        self.clf = SVC(class_weight='balanced', probability=True,)  # cache_size=500
        if self.is_grid_search:
            self._check_param_grid(self.param_grid)
        else:
            self.C = C
            self.gamma = gamma
            self.kernel = kernel

    def train(self, x_train, y_train):
        if self.is_grid_search:
            svm = self.grid_search(x_train, y_train)
        else:
            svm = self.clf.fit(x_train, y_train)
        return svm

    def grid_search(self, x_train, y_train):
        pipe = Pipeline([
            ('classify', self.clf)
        ])
        grid = GridSearchCV(pipe, cv=5, n_jobs=-1, param_grid=self.param_grid, iid=True)
        x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.4, random_state=1, shuffle=True)
        clf = grid.fit(x_train, y_train)
        C, gamma = clf.best_params_['classify__C'], clf.best_params_['classify__gamma'],
        self.best_score = clf.best_score_
        return clf.best_estimator_

    def _check_param_grid(self, param_grid):
        kernel = param_grid.get('kernel', 'rbf')
        C_start = param_grid.get('C_start', None)
        C_end = param_grid.get('C_end', None)
        g_start = param_grid.get('g_start', None)
        g_end = param_grid.get('g_end', None)
        param_ls = [C_start, C_end, g_start, g_end]
        if all([1 if i is not None else 0 for i in param_ls]):
            C_range = np.logspace(C_start, C_end, C_end - C_start + 1, base=2)
            gamma_range = np.logspace(g_start, g_end, g_end - g_end + 1, base=2)
            self.param_grid = [{'kernel': [kernel], 'C': C_range, 'gamma': gamma_range}]
        else:
            print('lost parmameters, will use default parameters ')
            C_range = np.logspace(-5, 15, 21, base=2)  # 21
            gamma_range = np.logspace(-15, 3, 19, base=2)  # 19
            # self.param_grid = [{'kernel': [kernel], 'C': C_range, 'gamma': gamma_range}]
            self.param_grid = [{'classify__kernel': [kernel], 'classify__C': C_range, 'classify__gamma': gamma_range}]


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
