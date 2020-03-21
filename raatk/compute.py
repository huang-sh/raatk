import os
from pathlib import Path
from functools import partial
from concurrent import futures

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold

try:
    from . import classify as al
    from . import utils as ul
except ImportError:
    import classify as al
    import utils as ul
 

np.seterr(all='ignore')


def grid_search(x, y, param_grid):
    grid = GridSearchCV(SVC(random_state=1), cv=5, n_jobs=-1, param_grid=param_grid)
    clf = grid.fit(x, y)
    C, gamma = clf.best_params_['C'], clf.best_params_['gamma']
    return C, gamma
    
def train(x, y, C, gamma, probability=False):
    svc = SVC(class_weight='balanced', C=C, gamma=gamma, random_state=1, probability=probability)
    model = svc.fit(x, y)
    return model

def save_model(model, file_path):
    joblib.dump(model, file_path)

def batch_train(in_dir, out_dir, C, gamma, n_job):
    def process_func(file, C, gamma):
        x, y = ul.load_data(file, normal=True)
        model = train(x, y, C, gamma)
        return model
    train_func = partial(process_func, C=C, gamma=gamma)
    dirs = Path(in_dir)
    out_dir = Path(out_dir)
    with Parallel(n_jobs=n_job) as train_pa, Parallel(n_jobs=int(n_job/2),
                                                       backend='threading') as save_pa:
        for type_ in dirs.iterdir():
            models = train_pa(delayed(train_func)(file) for file in type_.iterdir())
            type_name = type_.name
            type_dir = out_dir / type_name
            type_dir.mkdir(exist_ok=True)
            save_pa(delayed(save_model)(m, type_dir/ f.stem) for m,f in zip(models, type_.iterdir()))

def predict(x, model):
    y_pred = model.predict(x)
    try:
        y_prob = model.predict_proba(x)  ## 模型训练 probability=True才能用
    except:
        y_prob = None
    return y_pred, y_prob

# TODO 分离clf？
def evaluate(x, y, cv, C=None, gamma=None, clf=None, probability=False):
    if clf is None:
        clf = SVC(class_weight='balanced', C=C, gamma=gamma,
                 random_state=1, probability=probability)
    evalor = al.Evaluate(clf, x, y, probability=probability)
    k = int(cv)
    if k == -1:
        evalor.loo() 
        metric_dic = evalor.get_eval_idx()
    elif int(k) > 1:
        evalor.kfold(k)
        metric_dic = evalor.get_eval_idx()
    else:
        evalor.holdout(k)
        metric_dic = evalor.get_eval_idx()
    return metric_dic 

def batch_evaluate(in_dir, out_dir, cv, C, gamma, n_job):
    
    def eval_(file, C, gamma, cv):
        x, y = ul.load_data(file, normal=True)
        metric_dic = evaluate(x, y, cv, C=C, gamma=gamma)
        oa = sum([sum(i[:, 1, 1]) for i in metric_dic['mcm']]) / sum([len(i) for i in metric_dic['y_true']])
        OA = np.array([oa for _ in np.unique(metric_dic['y_true'][0])])
        for i in range(len(metric_dic['y_true'])):
            metric_dic["sub_metric"][i].append(OA)
        return metric_dic["sub_metric"]
        
    eval_func = partial(eval_, cv=cv, C=C, gamma=gamma)
    all_metric_dic = {}
    with Parallel(n_jobs=n_job) as eval_pa:
        for type_ in in_dir.iterdir():
            metrics_iter = eval_pa(delayed(eval_func)(file) for file in type_.iterdir())
            all_metric_dic[type_] = [i for i in metrics_iter]
    return all_metric_dic

def feature_select(x, y, C, gamma, step, cv, n_jobs):
    scaler = Normalizer()
    scaler_ft = partial(scaler.fit_transform, y=y)
    selector = VarianceThreshold()
    new_x = selector.fit_transform(x)
    score_idx = selector.get_support(indices=True)
    sb = SelectKBest(k='all')
    new_data = sb.fit_transform(new_x, y)
    f_value = sb.scores_
    idx_score = [(i, v) for i, v in zip(score_idx, f_value)]
    rank_score = sorted(idx_score, key=lambda x: x[1], reverse=True)
    feature_idx = [i[0] for i in rank_score]
    evla_func =  partial(evaluate, y=y, C=C, gamma=gamma, cv=cv)
    feature_len = len(feature_idx)
    feature_len // step
    if feature_len // step * step == feature_len:
        step_num = feature_len // step + 1
    else:
        step_num = feature_len // step + 2
    result_ls = Parallel(n_jobs=n_jobs)(
        delayed(evla_func)(
            scaler_ft(x[:, feature_idx[:i*step]])) for i in range(1, step_num))
    return result_ls, feature_idx
