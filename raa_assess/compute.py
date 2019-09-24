import os
import json
from functools import partial
from concurrent import futures


import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectKBest, VarianceThreshold

try:
    from . import classify as al
    from . import utils as ul
except ImportError:
    import classify as al
    import utils as ul
 

HPOD = 0.6

def model_hpo(x, y, **kwargs):
    c = kwargs.get("c", None)
    g = kwargs.get("gamma", None)
    if c and g:
        model = al.SvmClassifier(grid_search=False, C=c, gamma=g)
    else:
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

def process_eval_func(file, cv=-1, hpod=.6, **kwargs): 
    c = kwargs.get("c", None)
    g = kwargs.get("gamma", None)
    hpo_x, hpo_y = ul.data_to_hpo(file, hpod=hpod)
    if c and g:
        clf = model_hpo(hpo_x, hpo_y, **kwargs)
    else:
        clf = model_hpo(hpo_x, hpo_y)
    eval_x, eval_y = ul.load_normal_data(file)
    metrics = evaluate(clf, eval_x, eval_y, cv=cv)
    return metrics

def para_search(naa_path, hpod=.6):
    hpo_x, hpo_y = ul.data_to_hpo(naa_path, hpod=hpod)
    clf = model_hpo(hpo_x, hpo_y)
    para = clf.get_params()
    c = para["classify__C"]
    gamma = para["classify__gamma"]
    return c, gamma

def all_eval(n_fea_dir, result_path, n, cv, hpod, cpu):
    to_do_map = {}
    result_dic = {}
    max_work = min(cpu, os.cpu_count())
    naa_path = os.path.join(n_fea_dir, f'20_{n}n.csv')
    c, gamma = para_search(naa_path, hpod=hpod)
    para = {"c": c, "gamma": gamma}
    with futures.ProcessPoolExecutor(int(max_work)) as pp:
        evla_func = partial(process_eval_func, cv=cv, hpod=hpod, **para)
        for type_dir, file_ls in ul.parse_path(n_fea_dir, filter_format='csv'):
            for file in file_ls:
                file_path = os.path.join(type_dir, file)
                future = pp.submit(evla_func, file_path)
                type_num = os.path.basename(type_dir)
                to_do_map[future] = [type_num, f"{file.split('_')[0]}"]
        else:
            if os.path.exists(naa_path):
                future = pp.submit(evla_func, naa_path)
                to_do_map[future] = ['natural amino acids', '20s']
        done_iter = futures.as_completed(to_do_map)
        for it in done_iter:
            info = to_do_map[it]
            metric, _ = it.result()
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
            result_dic[t]['20'] = naa_dic
    print(f"k : {n}, c : {c}, gamma : {gamma}")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_dic, f, indent=4)


def feature_select(feature_file, cpu, cv=-1, hpod=.6):
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
    evla_func = partial(process_eval_func, cv=cv, hpod=hpod)
    results = Parallel(n_jobs=cpu)(
        delayed(evla_func)((X[:, feature_idx[:i+1]], y)) for i, idx in enumerate(feature_idx))
    acc_ls = []
    for i, it in enumerate(results, 1):
        acc, *_ = it[0]
        print(f"{i:<2} ----> {acc[0]}")
        acc_ls.append([acc[0], i])
    return [i[0] for i in acc_ls]

def feature_mix(files, cv=-1, hpod=0.6):
    data_ls = [np.genfromtxt(file, delimiter=',')[:, 1:] for file in files]
    mix_data = np.hstack(data_ls)
    y = np.genfromtxt(files[0], delimiter=',')[:, 0]
    x = mix_data
    acc_ls = feature_select((x, y), cv=cv, hpod=hpod)
    return acc_ls

def own_func(file_ls, feature_file, cluster, n, hpod, **kwargs):
    ul.one_file(file_ls, feature_file, cluster, n, idx=len(cluster))
    metrics, cm = process_eval_func(feature_file, cv=-1, hpod=hpod, **kwargs)
    return metrics, cm
