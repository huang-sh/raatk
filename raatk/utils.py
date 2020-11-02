import os
import csv
import json
import sqlite3
from pathlib import Path
from itertools import chain
from functools import partial
from itertools import product
from concurrent import futures

import joblib
import numpy as np
import numpy.lib.recfunctions as rf 
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis


try:
    from . import draw
    from . import feature as fea
    from . import metrics as mt
except ImportError:
    import draw
    import feature as fea
    import metrics as mt



BASE_PATH = Path(__file__).parent
RAA_DB = BASE_PATH / 'nr_raa_data.db'
NAA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
       'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


svm_params = SVC._get_param_names()
knn_params = KNeighborsClassifier._get_param_names()
rf_params = RandomForestClassifier._get_param_names()
clf_param_names = {'svm': svm_params, 'knn': knn_params, 'rf': rf_params}
clf_dic = {'svm': SVC, 'knn': KNeighborsClassifier, 'rf': RandomForestClassifier}


def check_aa(aa):
    if aa[0] == "-" or aa[-1] == "-":
        raise ValueError("amino acid cluster is wrong!")
    if "-" not in aa:
        raise ValueError("need an amino acid cluster!")
    aa = aa.strip().upper()
    cls_ls = list(aa.replace("-", "")).sort()
    if NAA.sort() != cls_ls:
        raise ValueError("amino acid cluster is wrong!")
    
# TODO - query optimization
def reduce_query(type_id, size):
    conn = sqlite3.connect(str(RAA_DB))
    cursor = conn.cursor()
    cursor.execute('select r.type_id, c.size, c.scheme, r.method from raa r \
                inner join cluster c on r.type_id=c.type_id \
                where  c.size in (%s) and r.type_id in (%s)' % (size, type_id))
    raa_clusters = cursor.fetchall()
    cursor.close()
    conn.commit()
    conn.close()
    return raa_clusters

def read_fasta(seq):
    seq_ls = []
    for line in seq:
        line = line.strip()
        if not line:
            continue
        if line[0] == '>':
            if seq_ls:
                yield descr, ''.join(seq_ls)
                seq_ls.clear()
            descr = line
        else:
            seq_ls.append(line)
    else:
        yield descr, ''.join(seq_ls)


def reduce(seqs, aa, raa=None):
    """ reduce seq based on rr
    :param seqs: seq lines, iter
    :param aa: cluster aa, list or tuple
    :param raa: representative aa, list or tuple
    :return:
    """
    if not raa:
        raa = [i.strip()[0] for i in aa]
    for i, j in zip(aa, raa):
        if j not in i:
            raise ValueError(f'raa or clustered_aa is wrong!')
    aa_dic = dict(zip(raa, aa))
    for seq in seqs:
        descr, seq = seq
        for key, val in aa_dic.items():
            if key == val:
                continue
            else:
                for ele in val:
                    seq = seq.replace(ele, key)
        yield descr, seq
    
def reduce_to_file(file, aa, output,):
    with output.open("w") as wh:
        rh = open(file, "r")
        seqs = read_fasta(rh)
        reduced_seqs = reduce(seqs, aa)
        for descr, r_seq in reduced_seqs:
            wh.write(descr)
            wh.write("\n")
            for i in range(len(r_seq) // 80 + 1):
                wh.write(r_seq[i*80:(i+1)*80])
                wh.write("\n")
            else:
                wh.write(r_seq[(i+1)*80:])
        rh.close()

def batch_reduce(file, cluster_info, out_dir):
    with futures.ThreadPoolExecutor(len(cluster_info)) as tpe:
        to_do_map = {}
        for idx, item in enumerate(cluster_info, 1):
            type_id, size, cluster, _ = item
            aa = [i[0] for i in cluster.split("-")]
            type_n = "".join(["type",str(type_id)])
            reduce_name = '-'.join([str(size), "".join(aa)]) + '.txt'
            rfile = out_dir / type_n / reduce_name
            mkdirs(rfile.parent)
            aa = cluster.split('-')
            future = tpe.submit(reduce_to_file, file, aa, rfile)
            to_do_map[future] = type_id, size, cluster
        done_iter = futures.as_completed(to_do_map)
        for i in done_iter:
            type_id, size, cluster = to_do_map[i]
            print("done %s %s %s" % (type_id, size, cluster)) 

def extract_feature(feature_file, raa, k, gap, lam, count=False):
    with open(feature_file, "r") as rh:
        seqs = read_fasta(rh)
        fea_func = partial(fea.seq_aac, raa=raa, k=k, gap=gap, lam=lam, count=count)
        seq_vec = np.array([fea_func(sq[1]) for sq in seqs])
    return seq_vec
            
# TODO - IO optimization     
def batch_extract(in_dirs, out_dir, k, gap, lam, n_jobs=1, count=False):

    def parse_filepath(in_dirs, out_dir):
        tps_iter = [Path(i).iterdir() for i in in_dirs]
        for tps in zip(*tps_iter):
            type_dir = out_dir / tps[0].name
            type_dir.mkdir(exist_ok=True)
            sizes_iter = [size.iterdir() for size in tps]
            for sfs in zip(*sizes_iter):
                szie_stem = sfs[0].stem
                out = type_dir / (szie_stem + ".csv")
                raa_str = szie_stem.split("-")[-1]
                yield out, sfs, raa_str

    def files_extract(raa, k, gap, lam, count, *files):
        xy_ls = []
        for idx, file in enumerate(files):
            xy = extract_feature(file, raa, k, gap, lam, count=count)
            y = np.array([[idx]]*xy.shape[0])
            xy = np.hstack([y, xy])
            xy_ls.append(xy)
        new_xy = np.vstack(xy_ls)
        return new_xy

    def feature2file(out, files, raa, k, gap, lam, count):
        data = files_extract(raa, k, gap, lam, count, *files)
        aa_ls = [''.join(aa) for aa in product(raa, repeat=k)]
        aa_ls.insert(0, 'label')
        header = ','.join(aa_ls)
        write_array(out, data, header=header)

    extract_fun = partial(feature2file, k=k, gap=gap, lam=lam, count=count)
    with Parallel(n_jobs=n_jobs) as pl:
        pl(delayed(extract_fun)(*paras) for paras in parse_filepath(in_dirs, out_dir))

def dic2array(result_dic, key='OA', cls=0):
    acc_ls = []  # all type acc
    type_ls = [type_id for type_id in result_dic.keys()]
    type_ls.sort(key=lambda x: int(x[4:]))
    all_score_array = np.zeros([len(type_ls), 19])
    for idx, ti in enumerate(type_ls):
        type_ = result_dic[ti]
        score_size_ls = []
        for size in range(2, 21):
            key_scores = type_.get(str(size), {key: 0}).get(key, 0)
            score = key_scores[cls] if isinstance(key_scores, list) else key_scores
            score_size_ls.append(score)
        all_score_array[idx] = score_size_ls
    return all_score_array, type_ls

def filter_type(score_metric, type_ls, filter_num=8):
    filter_scores, filter_type = [], []
    for type_score, type_id in zip(score_metric, type_ls):
        scores = [i for i in type_score if i > 0]
        if len(scores) >= 8:
            filter_scores.append(type_score)
            filter_type.append(type_id)
    return np.array(filter_scores), filter_type


def heatmap_txt(data, types, out):
    with open(out, "w", newline="") as f:
        fc = csv.writer(f)
        col = ["size"] + types
        fc.writerow(col)
        for row, size in zip(data, range(2, 21)):
            line = [size] + row.tolist()
            fc.writerow(line)


def eval_plot(result_dic, out, fmt, filter_num=8):
    key = 'acc'
    scores, types = dic2array(result_dic, key=key)
    f_scores, f_types = filter_type(scores, types, filter_num=filter_num)
    
    heatmap_path = out / f'{key}_heatmap.{fmt}'
    txt_path = out / f'{key}_heatmap.csv'
    heatmap_txt(scores.T, types, txt_path)
    hmd = scores.T * 100
    fhmd = f_scores.T * 100
    vmin = np.min(hmd[hmd>0]) // 10 * 10
    vmax = np.max(hmd[hmd>0]) // 10 * 10 + 10
    draw.p_acc_heat(hmd, vmin, vmax, types, heatmap_path)
    f_heatmap_path = out / f'f{filter_num}_{key}_heatmap.{fmt}'
    draw.p_acc_heat(fhmd, vmin, vmax, f_types, f_heatmap_path)
    
    f_scores_arr = f_scores[f_scores > 0]
    size_arr = np.array([np.arange(2, 21)] * f_scores.shape[0])[f_scores > 0]
    acc_size_path = out / f'acc_size_density.{fmt}'
    acc_path = out / f'acc_density.{fmt}'
    size_arr = size_arr.flatten()
    draw.p_bivariate_density(size_arr, f_scores_arr, acc_size_path)
    draw.p_univariate_density(f_scores_arr*100, out=acc_path)

    max_type_idx_arr, max_size_idx_arr = np.where(f_scores == f_scores.max())
    m_type_idx, m_size_idx = max_type_idx_arr[0], max_size_idx_arr[0]  # 默认第一个
    cp_path = out / f'acc_comparsion.{fmt}'
    diff_size = f_scores[m_type_idx]
    same_size = f_scores[:, m_size_idx]
    types_label = [int(i[4:]) for i in f_types]
    draw.p_comparison_type(diff_size, same_size, types_label, cp_path)
    return f_scores_arr


def parse_path(feature_folder, filter_format='csv'):
    """
    :param feature_folder: all type feature folder path
    :return:
    """
    path = os.walk(feature_folder)
    for root, dirs, file in path:
        if root == feature_folder:
            continue
        yield root, [i for i in file if i.endswith(filter_format)]


def mkdirs(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

def load_structured_data(file): ## file for data (x,y)
    if Path(str(file)).is_file():
        structured_data = np.genfromtxt(file, delimiter=',', names=True, dtype=float)
        data = rf.structured_to_unstructured(rf.repack_fields(structured_data)) 
    else:
        raise FileNotFoundError(file) # raise error
    data = data.reshape(1, -1) if len(data.shape) == 1 else data
    names = structured_data.dtype.names
    return names, data 

def normal_data(data):
    scaler = Normalizer()
    new_data = scaler.fit_transform(data)
    return new_data

def load_data(file, label_exist=True, normal=True):
    names, data = load_structured_data(file)
    if label_exist:
        x, y = data[:, 1:], data[:, 0]
    else:
        x, y = data, None
    if normal:
        x = normal_data(x)
    return names, (x, y)


def load_model(model_path):
    model = joblib.load(model_path)
    return model

def merge_feature_file(label, file):
    if label is None:
        data_ls = [load_structured_data(file)[1] for file in file]
        mix_data = np.vstack(data_ls)
    else:
        data_ls = []
        for idx, f in zip(label, file):
            data = load_structured_data(f)[1]
            data[:, 0] = idx
            data_ls.append(data)
        mix_data = np.vstack(data_ls)
    return mix_data

def write_array(file, *data, header=None):
    data = np.hstack(data)
    if header:
        np.savetxt(file, data, delimiter=",", fmt="%.6f", header=header, comments='')
    else:
        np.savetxt(file, data, delimiter=",", fmt="%.6f")
    
def split_data(file, test_size):
    names, data = load_structured_data(file)
    data_train, data_test = train_test_split(data, test_size=test_size, random_state=1)
    return data_train, data_test, names
    
def param_grid(c, g, kernel):
    c_range = np.logspace(*c, base=2)
    gamma_range = np.logspace(*g, base=2)
    params = [{'kernel': kernel, 'C': c_range,
                'gamma': gamma_range}]
    return params

def mean_metric(cv_metric_dic):
    cv = len(cv_metric_dic)
    acc, mcc = 0, 0
    precision, recall, f1_score = 0, 0, 0
    for fold, metric in cv_metric_dic.items():
        precision = np.add(precision, metric["precision"]/cv)
        recall = np.add(recall, metric["recall"]/cv)
        f1_score = np.add(f1_score, metric["f1-score"]/cv)
        acc = np.add(acc, metric["acc"]/cv)
        mcc = np.add(mcc, metric["mcc"]/cv)
    metric_dic = {
        'precision': precision.tolist(),
        'recall': recall.tolist(), 'mcc': mcc.tolist(),
        'acc': acc.tolist(),'f1-score': f1_score.tolist()}
    return metric_dic

def metric_dict2json(all_metric_dic, path):
    result_dic = {}
    naa_metric = {}
    for type_dir, cv_metric_ls in all_metric_dic.items():
        result_dic.setdefault(type_dir.name, {})
        for size_dir, cv_metric_dic in zip(type_dir.iterdir(), cv_metric_ls):
            metric_dic = mean_metric(cv_metric_dic)
            size_ = size_dir.name.split("-")[0]
            result_dic[type_dir.name][size_] = metric_dic
    naa_metric = result_dic.pop('naa', {})
    if naa_metric:
        [result_dic[tk].update(naa_metric) for tk in result_dic]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result_dic, f, indent=4)

def roc_auc_save(clf, x, y, cv, fmt, out):
    if cv is None:
        viz = mt.roc_curve_plot(clf, x, y)
    elif cv in (1, -1):
        viz = mt.loo_roc_curve_plot(clf, x, y)
    else:
        viz = mt.cv_roc_curve_plot(clf, x, y, cv)
    if fmt == 'txt':
        fpr = viz.fpr.reshape(-1, 1)
        tpr = viz.fpr.reshape(-1, 1)
        auc = viz.roc_auc
        write_array(f"{out}-{auc:.4f}.csv", fpr, tpr)
    else:
        plt.savefig(f"{out}.{fmt}", dpi=1000)


TEXT = """
    敏感度(Sensitivity, SN)也称召回率(Recall, RE):	
            Sn = Recall = TP / (TP + FN)
    特异性(Specificity, SP):
            Sp = TN / (TN + FP)
    精确率(Precision, PR)也称阳极预测值(Positive Predictive Value, PPV):	
            Precision= PPV = TP / (TP + FP)
    预测成功率(Accuracy, Acc):
            Acc = (TP + TN) / (TP + FP + TN + FN)
    Matthew 相关系数(Matthew's correlation coefficient, Mcc):
        MCC = (TP*TN- FP*FN)/sqrt((TP + FP)*(TN + FN)*(TP + FN)*(TN + FP)).其中sqrt代表开平方.
"""
              
def save_report(metric_dic, report_file):
    with open(report_file, "w", encoding="utf-8") as f:
        head = (f"     {'tp':^5}{'fn':^5}{'fp':^5}{'tn':^5}" +
                f"{'recall':^8}{'precision':^11}{'f1-score':^11}\n")
        for idx, sm in metric_dic.items():
            f.write(f"{idx:^50}")
            f.write('\n')
            for i, j in enumerate(sm['cm']):
                f.write(f"{i:<4}")
                f.write('  '.join(map(str, j.tolist())))
                f.write('\n')
            f.write('\n')
            f.write(head)
            tp, fn, fp, tn = sm['tp'], sm['fn'], sm['fp'], sm['tn']
            ppr, recall, f1s = sm['precision'], sm['recall'], sm['f1-score']
            cls_i = "{:^5}{:^5}{:^5}{:^5}{:^5}{:^8.2f}{:^11.2f}{:^11.2f}\n"
            acc_i = "acc{:>48.2f}\n"
            mcc_i = "mcc{:>48.2f}"
            for i in range(len(tp)):
                line = (i, tp[i], fn[i], fp[i], tn[i], recall[i], ppr[i], f1s[i])
                f.write(cls_i.format(*line))
            f.write(acc_i.format(sm['acc']))
            f.write(mcc_i.format(sm['mcc']))
            f.write("\n")
            f.write("-"*55)
            f.write("\n")
        f.write("\n")

def filter_args(kwargs, clf_params):
    params = *kwargs,
    target_params = set(params) & set(clf_params)
    param_kic = {}
    for p in target_params:
        if isinstance(kwargs[p], (int, float)):
            param_kic[p] = kwargs[p]
        elif kwargs[p].replace(".",'').isnumeric():
            param_kic[p] = float(kwargs[p])
        else:
            param_kic[p] = kwargs[p]    
    return param_kic

def select_clf(args):
    kwargs = dict(args._get_kwargs())
    param_dic = filter_args(kwargs, clf_param_names[args.clf])
    clf = clf_dic[args.clf](**param_dic)
    return clf

def exist_file(*file_path):
    for file in file_path:
        f = Path(file)
        if f.is_file():
            pass
        else:
            print("File not found!")
            exit()

#TODO add other method
def feature_reduction(x, y, n_components=2):
    from sklearn.pipeline import make_pipeline
    nca = make_pipeline(Normalizer(),
            NeighborhoodComponentsAnalysis(init='auto',
                            n_components=n_components, random_state=1))
    rx = nca.fit_transform(x,y)
    return rx, y

def fmt_transfer(file, fmt='arff'):
    names, data = load_structured_data(file)
    out = file.with_suffix(f'.{fmt}')
    with open(out, 'w', newline='\n') as f:
        fc = csv.writer(f)
        fc.writerow([r'% generated by RAATK'])
        fc.writerow(['@Relation Protein'])
        for aa in names[1:]:
            fc.writerow([f'@attribute {aa} NUMERIC'])
        labels = ','.join([f'class{int(i)}' for i in np.unique(data[:, 0])])
        classes = ''.join(['{', labels, '}'])
        f.write(f"@attribute class {classes}\n")
        fc.writerow(["@Data"])
        for row in data:
            row = row.tolist()
            line = row[1:] + [f'class{int(row[0])}']
            fc.writerow(line)

def cluster_link(source, target):
    sl, tl, vl = [], [], []
    for ti, taac in enumerate(target):
        taa_set = set(taac)
        aac_len = len(taac)
        for si, saac in enumerate(source):
            intersect = taa_set & set(saac)
            if intersect:
                sl.append(si)
                tl.append(ti)
                vl.append(len(intersect))
                aac_len -= len(intersect)
            if aac_len == 0:
                break
    return sl, tl, vl 

def type_link(clusters):
    base_idx = 0    
    source_idx, target_idx, values = [], [], []
    for i in range(len(clusters)-1):
        sl, tl, vl = cluster_link(clusters[i], clusters[i+1])
        sidx = [i+base_idx for i in sl]
        base_idx += len(clusters[i])
        tidx = [i+base_idx for i in tl]
        source_idx.extend(sidx)
        target_idx.extend(tidx)
        values.extend(vl)
    return source_idx, target_idx, values

def plot_sankey(out, clusters, title):
    source_idx, target_idx, values = type_link(clusters)
    labels = list(chain(*clusters))
    draw.sankey(labels, source_idx, target_idx, values, out, title)