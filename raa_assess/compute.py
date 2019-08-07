import os
import json
from functools import partial
from concurrent import futures

import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold

from . import classify as al
from . import utils as ul


def model_hpo(x, y):
    model = al.SvmClassifier()
    clf = model.train(x, y)
    return clf
    
def evaluate(clf, x, y, cv=-1, **kwargs):
    evalor = al.Evaluate(clf, x, y)
    k = int(cv)
    if k == -1:
        metrics = evalor.loo() ## to do
    elif int(k):
        metrics = evalor.kfold(k)
    else:
        metrics = evalor.holdout(k)
    return metrics

def process_eval_func(file, cv=-1, hpo=1): # 
    hpo_x, hpo_y = ul.data_to_hpo(file, hpo=1)
    clf = model_hpo(hpo_x, hpo_y)
    eval_x, eval_y = ul.load_normal_data(file)
    metrics = evaluate(clf, eval_x, eval_y, cv=-1)
    return metrics

def all_eval(folder_n, result_path, n, cv, hpo, cpu):
    to_do_map = {}
    result_dic = {}
    max_work = min(cpu, os.cpu_count())
    with futures.ProcessPoolExecutor(int(max_work)) as pp:
        evla_func = partial(process_eval_func, cv=cv, hpo=hpo)
        for type_dir, file_ls in ul.parse_path(folder_n, filter_format='csv'):
            for file in file_ls:
                file_path = os.path.join(type_dir, file)
                future = pp.submit(evla_func, file_path)
                type_num = os.path.basename(type_dir)
                to_do_map[future] = [type_num, f"{file.split('_')[0]}"]
        else:
            naa_path = os.path.join(folder_n, f'20_{n}n.csv')
            if os.path.exists(naa_path):
                future = pp.submit(evla_func, naa_path)
                to_do_map[future] = ['natural amino acids', '20s']
        done_iter = futures.as_completed(to_do_map)
        for it in done_iter:
            info = to_do_map[it]
            metric = it.result()
            acc, sn, sp, ppv, mcc = metric[0]
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

def feature_select(feature_file, cv=-1, hpo=1):
    X, y = ul.load_normal_data(feature_file)
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
        evla_func = partial(process_eval_func, cv=cv, hpo=hpo)
        for i, idx in enumerate(feature_idx):
            index = feature_idx[:i+1]
            x = X[:, index]
            print(x.shape)
            future = pp.submit(evla_func, (x, y))
            to_do_map[future] = [i+1, rank_score[i]]
        done_iter = futures.as_completed(to_do_map)
        acc_ls = []
        for it in done_iter:
            idx, score = to_do_map[it]
            acc, *_ = it.result()[0]
            acc_ls.append(acc[0])
        acc_ls.sort()
    return acc_ls

def feature_mix(files, cv=-1, hpo=1):
    data_ls = [np.genfromtxt(file, delimiter=',')[1:] for file in files]
    mix_data = np.hstack(data_ls)
    x = mix_data[:, 1:]
    y = mix_data[:, 0]
    acc_ls = feature_select((x, y), cv=-1, hpo=1)
    return acc_ls

def own_func(file_ls, feature_file, cluster, n):
    ul.one_file(file_ls, feature_file, cluster, n, idx=len(cluster))
    metrics, cm = process_eval_func(feature_file, cv=-1, hpo=1)
    return metrics, cm
