from pathlib import Path
from functools import partial

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold

try:
    from . import utils as ul
    from . import metrics as mt
except ImportError:
    import utils as ul
    import metrics as mt
 

np.seterr(all='ignore')


def grid_search(x, y, param_grid, n_jobs):
    grid = GridSearchCV(SVC(random_state=1), cv=5, n_jobs=n_jobs, param_grid=param_grid)
    clf = grid.fit(x, y)
    C, gamma = clf.best_params_['C'], clf.best_params_['gamma']
    kernel = clf.best_params_['kernel']
    return C, gamma, kernel


def train(x, y, clf, out):
    model = clf.fit(x, y)
    joblib.dump(model, out)


# its discarded
def batch_train(in_dir, out_dir, C, gamma, n_job):

    def process_func(file, C, gamma):
        _, (x, y) = ul.load_data(file, normal=True)
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
        y_prob = model.predict_proba(x)  # 模型训练 probability=True才能用
    except:
        y_prob = None
    return y_pred, y_prob


def cv_predict(clf, x, y, cver, method="predict"):
    y_pred = cross_val_predict(clf, x, y, cv=cver, method=method)
    return y_pred


def evaluate(x, y, cv, clf):
    k = int(cv)
    if k in (-1, 1):
        metric_dic = mt.loo_metrics(clf, x, y)
    elif int(k) > 1:
        metric_dic = mt.cv_metrics(clf, x, y, cv)
    else:  ## hold out
        pass
    return metric_dic 


def batch_evaluate(in_dir, out_dir, cv, clf, n_job):
    
    def eval_(file, cv, clf):
        _, (x, y) = ul.load_data(file)
        metric_dic = evaluate(x, y, cv, clf)
        return metric_dic
        
    eval_func = partial(eval_, cv=cv, clf=clf)
    all_metric_dic = {}
    with Parallel(n_jobs=n_job) as eval_pa:
        for type_ in in_dir.iterdir():
            metrics_iter = eval_pa(delayed(eval_func)(file) for file in type_.iterdir())
            all_metric_dic[type_] = [i for i in metrics_iter]
    return all_metric_dic


def feature_select(x, y, step, cv, clf, n_jobs):
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
    evla_func =  partial(evaluate, y=y, cv=cv, clf=clf)
    feature_num = len(feature_idx)
    step_num = feature_num // step
    if step_num * step == feature_num:
        step_num = step_num + 1
    else:
        step_num = step_num + 2
    result_ls = Parallel(n_jobs=n_jobs)(
        delayed(evla_func)(
            scaler_ft(x[:, feature_idx[:i*step]])) for i in range(1, step_num))
    return result_ls, feature_idx
