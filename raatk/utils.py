import os
import csv
import json
import sqlite3
from pathlib import Path
from functools import partial
from concurrent import futures

import joblib
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, RocCurveDisplay, auc, plot_roc_curve

try:
    from . import draw
    from . import feature as fea
except ImportError:
    import draw
    import feature as fea


BASE_PATH = os.path.dirname(__file__)
RAA_DB = os.path.join(BASE_PATH, 'nr_raa_data.db')
NAA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
       'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


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
    conn = sqlite3.connect(RAA_DB)
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
        h = csv.writer(wh)
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

def extract_feature(feature_file, raa, k, gap, lam):
    with open(feature_file, "r") as rh:
        seqs = read_fasta(rh)
        fea_func = partial(fea.seq_aac, raa=raa, k=k, gap=gap, lam=lam)
        seq_vec = np.array([fea_func(sq[1]) for sq in seqs])
    return seq_vec
            
# TODO - IO optimization     
def batch_extract(in_dirs, out_dir, k, gap, lam, n_jobs=1):

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

    def files_extract(raa, k, gap, lam, *files):
        xy_ls = []
        for idx, file in enumerate(files):
            xy = extract_feature(file, raa, k, gap, lam)
            y = np.array([[idx]]*xy.shape[0])
            xy = np.hstack([y, xy])
            xy_ls.append(xy)
        new_xy = np.vstack(xy_ls)
        return new_xy

    def feature2file(out, files, raa, k, gap, lam):
        data = files_extract(raa, k, gap, lam, *files)
        np.savetxt(out, data, delimiter=",", fmt="%.6f")

    extract_fun = partial(feature2file, k=k, gap=gap, lam=lam)
    with Parallel(n_jobs=n_jobs) as pl:
        pl(delayed(extract_fun)(*paras) for paras in parse_filepath(in_dirs, out_dir))

def roc_eval(x, y, model, out):
    svc_disp = plot_roc_curve(model, x, y)
    plt.savefig(out, dpi=1000, bbox_inches="tight")

def dic2array(result_dic, key='OA', cls=0):
    acc_ls = []  # all type acc
    type_ls = [type_id for type_id in result_dic.keys() if type_id != "naa"]
    type_ls.sort(key=lambda x: int(x[4:]))
    all_score_array = np.zeros([len(type_ls), 19])
    for idx, ti in enumerate(type_ls):
        type_ = result_dic[ti]
        score_size_ls = []
        for size in range(2, 21):
            key_scores = type_.get(str(size), {key: [0]*10}).get(key, 0)
            score = key_scores[cls] if key_scores else 0  
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
    key = 'OA'
    scores, types = dic2array(result_dic, key=key)
    f_scores, f_types = filter_type(scores, types, filter_num=filter_num)
    
    annot_size, tick_size, label_size = heatmap_font_size(scores.shape[1])
    f_annot_size, f_tick_size, f_label_size = heatmap_font_size(f_scores.shape[1])
    font_size = {"annot_size": annot_size, "tick_size": tick_size, "label_size": label_size}
    f_font_size = {"annot_size": f_annot_size, "tick_size": f_tick_size, "label_size": f_label_size}

    heatmap_path = out / f'{key}_heatmap.{fmt}'
    txt_path = out / f'{key}_heatmap.csv'
    heatmap_txt(scores.T, types, txt_path)
    draw.p_acc_heat(scores.T, 0.6, 1, types, heatmap_path, **font_size)
    f_heatmap_path = out / f'f{filter_num}_{key}_heatmap.{fmt}'
    draw.p_acc_heat(f_scores.T, 0.6, 1, f_types, f_heatmap_path, **f_font_size)
    
    f_scores_arr = f_scores[f_scores > 0]
    size_arr = np.array([np.arange(2, 21)] * f_scores.shape[0])[f_scores > 0]
    acc_size_path = out / f'acc_size_density.{fmt}'
    acc_path = out / f'acc_density.{fmt}'
    size_arr = size_arr.flatten()
    draw.p_bivariate_density(size_arr, f_scores_arr, acc_size_path)
    draw.p_univariate_density(f_scores_arr*100, acc_path)

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

def load_data(file_data, label_exist=True, normal=False): ## file for data (x,y)
    if isinstance(file_data, (tuple, list)):
        if all([isinstance(i, np.ndarray) for i in file_data]):
            x, y = file_data
    elif os.path.isfile(str(file_data)):
        data = np.genfromtxt(file_data, delimiter=',')
        if label_exist:
            x, y = data[:, 1:], data[:, 0]
        else:
            x, y = data, None
    else:
        raise FileNotFoundError(file_data) # raise error
    x = x.reshape(1, -1) if len(x.shape) == 1 else x
    if normal:
        scaler = Normalizer()
        x = scaler.fit_transform(x)
    return x, y

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def feature_mix(files):
    data_ls = [np.genfromtxt(file, delimiter=',')[:, 1:] for file in files]
    mix_data = np.hstack(data_ls)
    y = np.genfromtxt(files[0], delimiter=',')[:, 0]
    x = mix_data
    return x, y

def merge_feature_file(label, file):
    if label is None:
        data_ls = [np.genfromtxt(file, delimiter=',') for file in file]
        mix_data = np.vstack(data_ls)
    else:
        data_ls = []
        for idx, f in zip(label, file):
            data = np.genfromtxt(f, delimiter=',')
            data[:, 0] = idx
            data_ls.append(data)
        mix_data = np.vstack(data_ls)
    return mix_data

def write_array(file, *data):
    data = np.hstack(data)
    np.savetxt(file, data, delimiter=",", fmt="%.6f")
    
def split_data(file, test_size):
    data = np.genfromtxt(file, delimiter=',')
    data_train, data_test = train_test_split(data, test_size=test_size, random_state=1)
    return data_train, data_test
    
def param_grid(c, g):
    c_range = np.logspace(*c, base=2)
    gamma_range = np.logspace(*g, base=2)
    params = [{'kernel': ["rbf"], 'C': c_range,
                    'gamma': gamma_range}]
    return params
    
def save_json(metric_dic, path):
    result_dic = {}
    naa_metric = {}
    for type_dir, metric_ls in metric_dic.items():
        result_dic.setdefault(type_dir.name, {})
        for size_dir, metrics in zip(type_dir.iterdir(), metric_ls):
            acc, sn, sp, ppv, mcc, oa = [i.tolist() for i in np.mean(metrics, axis=0)]
            metric_dic = {'sn': sn, 'sp': sp, 'ppv': ppv,
                         'acc': acc, 'mcc': mcc, 'OA': oa}
            if type_dir.name == "naa":
                naa_metric = metric_dic
            else:
                size_ = size_dir.name.split("-")[0]
                result_dic[type_dir.name][size_] = metric_dic
    for type_key in result_dic:
        result_dic[type_key]["20"] = naa_metric
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result_dic, f, indent=4)



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
    y_true = metric_dic['y_true']
    y_pre = metric_dic['y_pre']
    mcm = metric_dic['mcm']
    sub_metric = metric_dic['sub_metric']
    with open(report_file, "w", encoding="utf-8") as f:
        for mcm_idx, sub_mcm in enumerate(mcm):
            cm = confusion_matrix(y_true[mcm_idx], y_pre[mcm_idx])
            clses = cm.shape[0]
            col_cls = "".join(([f"{'':<4}"]+[f"{i:<4}" for i in range(clses)]))
            f.write(col_cls)
            f.write('\n')
            for idx, line in enumerate(cm):
                row_preds = "".join(([f'{idx:<4}']+[f"{i:<4}" for i in line]))
                f.write(row_preds)
                f.write('\n')
            f.write("\n")
            
            kfi = f"{mcm_idx}\n"
            f.write(kfi)
            col = f"     {'tp':<4}{'fn':<4}{'fp':<4}{'tn':<4}{'sn':<7}{'sp':<7}{'ppv':<7}{'acc':<7}{'mcc':<7}\n"
            f.write(col)
            for pos_idx, line in enumerate(sub_mcm):
                (tn, fp), (fn, tp) = line
                accl, snl, spl, ppvl, mccl = sub_metric[mcm_idx]
                acc, sn, sp, ppv, mcc = accl[pos_idx]*100, snl[pos_idx]*100, spl[pos_idx]*100, ppvl[pos_idx]*100, mccl[pos_idx]*100
                linei = f"{pos_idx:<5}{tp:<4}{fn:<4}{fp:<4}{tn:<4}{sn:<7.2f}{sp:<7.2f}{ppv:<7.2f}{acc:<7.2f}{mcc:<7.2f}\n"
                f.write(linei)  
            f.write("\n\n")
        else:
            mean_mcm = np.mean(mcm, axis=0)
            mean_metric = np.mean(sub_metric, axis=0)
            f.write("mean\n")
            col = f"     {'tp':<4}{'fn':<4}{'fp':<4}{'tn':<4}{'sn':<7}{'sp':<7}{'ppv':<7}{'acc':<7}{'mcc':<7}\n"
            f.write(col)
            for pos_idx, line in enumerate(mean_mcm):
                (tn, fp), (fn, tp) = line
                accl, snl, spl, ppvl, mccl = mean_metric
                acc, sn, sp, ppv, mcc = accl[pos_idx]*100, snl[pos_idx]*100, spl[pos_idx]*100, ppvl[pos_idx]*100, mccl[pos_idx]*100
                linei = f"{pos_idx:<5}{tp:<4.1f}{fn:<4.1f}{fp:<4.1f}{tn:<4.1f}{sn:<7.2f}{sp:<7.2f}{ppv:<7.2f}{acc:<7.2f}{mcc:<7.2f}\n"
                f.write(linei)  
            f.write("\n\n")
        f.write(TEXT)
        f.write("\n\n")
        for fold_idx, y_labels in enumerate(zip(y_true, y_pre)):
            kfi = f"{fold_idx}\n"
            f.write(kfi)
            for label in zip(*y_labels):
                f.write(",".join(map(str, label)))
                f.write("\n")
            f.write("\n\n")

def k_roc_curve_plot(y_true, y_prob, out):
    if len(np.unique(y_true[0])) != 2:
        exit()
    fig, ax = plt.subplots()
    k_fold = len(y_true)
    num = sum(map(len, y_true))
    mean_fpr = np.linspace(0, 1, num)
    tprs = []
    aucs = []
    for idx, (yt, yp) in enumerate(zip(y_true, y_prob)):
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc = auc(fpr, tpr)
        viz = RocCurveDisplay(fpr, tpr, roc_auc, 'SVM')
        if k_fold == 1:
            name = 'ROC'
        else:
            name = f'ROC fold {idx}'
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            tprs.append(interp_tpr)
            aucs.append(roc_auc)            
        viz.plot(ax=ax, name=name, alpha=0.3, lw=1)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
    if k_fold > 1:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Receiver operating characteristic example")
        ax.legend(loc="lower right")
    plt.savefig(out)

def exist_file(*file_path):
    for file in file_path:
        f = Path(file)
        if f.is_file():
            pass
        else:
            print("file not found!")
            exit()

# 先将就用            
def heatmap_font_size(types):
    if types <= 10:
        annot_size = 4
        tick_size = 4
        label_size = 5
    elif types <= 20:
        annot_size = 3
        tick_size = 3
        label_size = 4     
    elif types <= 30:
        annot_size = 2.5
        tick_size = 2.5
        label_size = 3.5 
    elif types <= 40:
        annot_size = 2.5
        tick_size = 2.5
        label_size = 3.5 
    elif types <= 50:
        annot_size = 1.5
        tick_size = 2.5
        label_size = 3.5 
    elif types <= 60:
        annot_size = 1.5
        tick_size = 2.5
        label_size = 3.5 
    else:
        annot_size = 1
        tick_size = 2
        label_size = 3 
    return annot_size, tick_size, label_size
