import os
import csv
import json
from functools import partial
from concurrent import futures

import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold
import joblib

import classify as al
import utils as ul


def load_normal_data(file):
    data = np.genfromtxt(file, delimiter=',')
    x = data[:, 1:]
    y = data[:, 0]
    scaler = Normalizer()
    x = scaler.fit_transform(x)
    return x, y
    
def evaluate(test_file, cv=-1, hpo=1, **kwargs):
    train_x, train_y = load_normal_data(test_file)
    if hpo < 1:
        train_x, _, train_y, _ = train_test_split(train_x, train_y, shuffle=True,
                                                random_state=1, test_size=1-hpo)
    model = al.SvmClassifier()
    clf = model.train(train_x, train_y) 
    
    test_x, test_y = load_normal_data(test_file)
    test_data = np.genfromtxt(test_file, delimiter=',')

    evalor = al.Evaluate(clf, test_x, test_y)
    k = int(cv)
    if k == -1:
        metrics = evalor.loo() ## to do
    elif int(k):
        metrics = evalor.kfold(k)
    else:
        metrics = evalor.holdout(k)
    return metrics

def all_eval(folder_name, result_path, n, cv, hpo, cpu):
    folder_n = folder_name
    to_do_map = {}
    result_dic = {}
    max_work = min(cpu, os.cpu_count())
    with futures.ProcessPoolExecutor(max_work) as pp:
        for type_dir, file_ls in ul.parse_path(folder_n, filter_format='csv'):
            for file in file_ls:
                model_path = os.path.join(type_dir, file)
                file_path = model_path.replace('model', 'csv')
                future = pp.submit(evaluate, file_path, cv=cv, hpo=hpo)
                type_num = os.path.basename(os.path.dirname(file_path))
                to_do_map[future] = [type_num, f"{file.split('_')[0]}"]
        else:
            naa_path = os.path.join(folder_n, f'20_{n}n.csv')
            if os.path.exists(naa_path):
                future = pp.submit(evaluate, naa_path, k=cv, hpo=hpo)
                to_do_map[future] = ['natural amino acids', '20s']
        done_iter = futures.as_completed(to_do_map)
        for it in done_iter:
            info = to_do_map[it]
            metric = it.result()
            acc, sn, sp, ppv, mcc = metric
            one_dic = {'sn': sn.tolist(), 'sp': sp.tolist(), 'ppv': ppv.tolist(),
                      'acc': acc.tolist(), 'mcc': mcc.tolist()}
            if info[-1] == '20s':
                naa_dic = one_dic
            else:
                Type, size = info
                result_dic.setdefault(Type, {})
                result_dic[Type][size] = one_dic
        for t in result_dic:
            print(t)
            result_dic[t]['20'] = naa_dic
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_dic, f)


def al_comparison(file_path,):
    """

    :param file_path: feature file path
    :return:
    """
    classifier = {'SVM': al.SvmClassifier, 'RF': al.RfClassifier, 'KNN': al.KnnClassifier}
    result_dic = {}
    for clf in classifier:
        model = drill(classifier[clf], file_path)
        metrics, auc = evaluate(model, file_path)
        result_dic[clf] = (*metrics[5:], auc) # sn, sp, presision, acc, mcc, fpr, tpr, auc
    return result_dic


def feature_select(feature_file):
    fs_data = np.genfromtxt(feature_file, delimiter=',')
    x = fs_data[:, 1:]
    y = fs_data[:, 0]
    scaler = Normalizer()
    X = scaler.fit_transform(x)
    selector = VarianceThreshold()
    new_x = selector.fit_transform(X)
    score_idx = selector.get_support(indices=True)
    sb = SelectKBest(k='all')
    new_data = sb.fit_transform(new_x, y)
    f_value = sb.scores_
    idx_score = [(i, v) for i, v in zip(score_idx, f_value)]
    rank_score = sorted(idx_score, key=lambda x: x[1], reverse=True)
    feature_idx = [i[0] for i in rank_score]
    with futures.ProcessPoolExecutor() as pp:
        to_do_map = {}
        clf = al.SvmClassifier()
        clf = partial(clf.train, y_train=y)
        # return [sn, sp, acc, mcc, y_pro, auc, fpr, tpr]
        for i, idx in enumerate(feature_idx):
            index = feature_idx[:i+1]
            x = X[:, index]
            print(x.shape)
            future = pp.submit(clf, x)
            to_do_map[future] = [i+1, rank_score[i]]
        done_iter = futures.as_completed(to_do_map)
        log_info = []
        for it in done_iter:
            idx, score = to_do_map[it]
            sn, sp, acc, mcc, _ = it.result()
            # acc_ls.append(acc[0])
            log_info.append([idx, acc[0], score])
        log_info.sort(key=lambda x: x[0])
        acc_ls = [i[1] for i in log_info]
    return acc_ls


def fea_fusion():
    data = np.genfromtxt('17_3n.csv', delimiter=',')
    data1 = np.genfromtxt('17_1n.csv', delimiter=',')
    data2 = np.genfromtxt('17_2n.csv', delimiter=',')
    data = np.hstack((data1, data2[:, 1:], data[:, 1:]))
    x = data[:, 1:]
    y = data[:, 0]
    cp.feature_select(x, y, 'isp123-all.csv', 'isp123.pdf')