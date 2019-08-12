import os
import csv
import re
import sqlite3
from itertools import product
from concurrent import futures

import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

try:
    from . import draw
except ImportError:
    import draw


BASE_PATH = os.path.dirname(__file__)
RAA_DB = os.path.join(BASE_PATH, 'raa_data.db')
NAA = ['A', 'G', 'S', 'T', 'R', 'Q', 'E', 'K', 'N', 'D',
    'C', 'H', 'I', 'L', 'M', 'V', 'F', 'Y', 'P', 'W']


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
    lines = []
    for line in seq:
        if not line:
            continue
        if line[0] == '>':
            if lines:
                yield title, ''.join(lines)
                lines = []
            title = line[1:].strip()
        else:
            line = line.replace(' ', '').replace('\r', '')
            lines.append(line.strip())
    else:
        yield title, ''.join(lines)


def reduce(seqs, aa, raa=None):
    """ reduce seq based on rr
    :param seqs: seq lines, iter
    :param aa: cluster aa, list or tuple
    :param raa: representative aa, list or tuple
    :return:
    """
    if not raa:
        raa = [i[0] for i in aa]
    for i, j in zip(aa, raa):
        if j not in i:
            raise ValueError(f'raa or clustered_aa is wrong!')
    aa_dic = dict(zip(raa, aa))
    for seq in seqs:
        title, seq = seq
        for key, val in aa_dic.items():
            if key == val:
                continue
            else:
                for ele in val:
                    seq = seq.replace(ele, key)
        yield title, seq


def seq_aac(seqs, raa, n=1):
    """ extract aac feature
    :param seqs: seq lines
    :param raa: representative aa, list
    :param n: k-mer, int
    :return:
    """
    aa = [''.join(aa) for aa in product(raa, repeat=n)]
    for seq in seqs:
        title, seq = seq
        aa_fre = []
        seq_len = len(seq) -n + 1
        for a in aa:
            reg = re.compile(f"(?={a})")
            num = len(reg.findall(seq))
            aa_fre.append(num)
        aa_fre = [seq.count(i)/seq_len for i in aa]  # (len(seq)-n+1)
        yield title, aa_fre

def one_file(file_list, file_path, aa, n, idx=None,raa=None):
    """ write feature vector to a file
    :param file_list: train file list, list
    :param file_path: one size file path, string
    :param aa: cluster aa, list or tuple
    :param n: k-mer, int
    :param idx: index of reduced scheme in a type,
    :param raa: representative aa, list or tuple
    :return:
    """
    
    if os.path.isdir(file_path):
        file_name = f'{idx}_{n}n.csv'
        file_path = os.path.join(file_path, file_name)
    elif os.path.isfile(file_path):
        file_path = file_path
    with open(file_path, 'w') as handle:
        h = csv.writer(handle)
        for idx, file in enumerate(file_list):
            f = open(file, 'r')
            seq = read_fasta(f)
            simple_seq = reduce(seq, aa, raa)
            if not raa:
                raa = [i[0] for i in aa]
            base_aac = seq_aac(simple_seq, raa, n)
            for a in base_aac:
                line0 = [v for v in a[1]]
                line1 = [idx] + line0
                h.writerow(line1)
            f.close()

def thread_func(file_list, folder_n, n, clusters):
    to_do_map = {}
    with futures.ThreadPoolExecutor(28) as tpe:
        for idx, item in enumerate(clusters):
            tpi, size, cluster, _ = item
            aa = cluster.split('-')
            aa = [i for i in aa if i]
            type_dir = os.path.join(folder_n, f"type{tpi}")
            mkdirs(type_dir)
            file_path = os.path.join(folder_n, f"type{tpi}", f"{size}_{n}n.csv")
            future = tpe.submit(one_file, file_list, file_path, aa, n, idx=size)
            to_do_map[future] = tpi, size, cluster
        done_iter = futures.as_completed(to_do_map)
        for i in done_iter:
            print(i)

def reduce_seq(file_list, folder_n, n, cluster_info, p):
    mkdirs(folder_n)
    to_do_map = {}
    cluster_per = []
    counts = len(cluster_info)
    max_work = min(p, os.cpu_count(), counts)
    max_work = max(1, max_work)
    per = int(counts / max_work)
    tmp = []
    with futures.ProcessPoolExecutor(max_work) as ppe:
        for idx, item in enumerate(cluster_info, 1):
            tpi, size, _, _ = item
            if tmp == [tpi, size]:
                continue
            else:
                tmp = [tpi, size]
            if idx % per == 0:
                clusters = cluster_per.copy()
                future = ppe.submit(thread_func, file_list, folder_n, n, clusters)
                to_do_map[future] = [idx-per, idx]
                cluster_per.clear()
            cluster_per.append(item)
        else:
            future = ppe.submit(thread_func, file_list, folder_n, n, cluster_per)
            to_do_map[future] = [idx-per, idx]
        naa_path = os.path.join(folder_n, f'20_{n}n.csv')
        future = ppe.submit(one_file, file_list, naa_path, NAA, n)
        to_do_map[future] = "20s"
        done_iter = futures.as_completed(to_do_map)
        for f in done_iter:
            idx = to_do_map[f]
            print(f'{n}n --> {idx}', 'has done!')

def dic2array(result_dic, key='acc', filter_num=0, cls=0):
    acc_ls = []  # all type acc
    filtered_type_acc = []
    type_ls = [type_id for type_id in result_dic.keys()]
    type_ls.sort(key=lambda x: int(x[4:]))
    all_score_array = np.zeros([len(type_ls), 19])

    filtered_type_ls = type_ls.copy()
    filtered_score_ls = []
    sum_raa = sum([len(result_dic[ti].keys()) for ti in type_ls])
    if sum_raa < 600:
        filter_num = 0
    for idx, ti in enumerate(type_ls):
        type_ = result_dic[ti]
        score_size_ls = []
        for size in range(2, 21):
            if str(size) in type_:
                score = type_[str(size)][key][cls]
            else:
                score = 0
            score_size_ls.append(score)
        all_score_array[idx] = score_size_ls
        if len(type_) < filter_num:
            filtered_type_ls.remove(ti)
            continue
        filtered_score_ls.append(score_size_ls)
    filtered_score_array = np.array(filtered_score_ls)
    all_score = (all_score_array, type_ls)
    filtered_score = (filtered_score_array, filtered_type_ls)
    return all_score, filtered_score

def eval_plot(result_dic, n, out, fmt='tiff', filter_num = 8):
    key = 'acc'
    all_score, filter_score = dic2array(result_dic, key=key, filter_num=filter_num)
    scores, types = all_score
    f_scores, f_types = filter_score
    f_heatmap_path = os.path.join(out, f'{key}_f{filter_num}-heatmap_{n}n.{fmt}')
    heatmap_path = os.path.join(out, f'{key}_heatmap_{n}n.{fmt}')
    draw.p_acc_heat(f_scores.T, 0.6, 1, f_types, f_heatmap_path)
    draw.p_acc_heat(scores.T, 0.6, 1, types, heatmap_path)

    f_scores_arr = f_scores[f_scores > 0]
    size_arr = np.array([np.arange(2, 21)] * f_scores.shape[0])[f_scores > 0]
    path = os.path.join(out, f'acc_size_density-{n}n.{fmt}')
    size_arr = size_arr.flatten()
    draw.p_bivariate_density(size_arr, f_scores_arr, n, path)

    max_type_idx_arr, max_size_idx_arr = np.where(f_scores == f_scores.max())
    m_type_idx, m_size_idx = max_type_idx_arr[0], max_size_idx_arr[0]  # 默认第一个

    cp_path = os.path.join(out, f'comparsion_{n}n.{fmt}')
    diff_size = f_scores[m_type_idx]
    same_size = f_scores[:, m_size_idx]
    types_label = [int(i[4:]) for i in f_types]
    draw.p_comparison_type(diff_size, same_size, types_label, cp_path)

    fea_folder = f'{out}_{n}n'
    type_id = f_types[m_type_idx]
    file_name = f"{m_size_idx+2}_{n}n.csv"
    max_acc_fea_file = os.path.join(fea_folder, type_id, file_name)
    # com_result = cp.al_comparison(max_acc_fea_file)
    # roc_path = os.path.join(out, f'{n}n_al_roc.{fmt}')
    # draw.p_roc_al(param, roc_path)
    return f_scores_arr, max_acc_fea_file

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

def load_normal_data(file_data): ## file for data (x,y)
    if os.path.isfile(str(file_data)):
        data = np.genfromtxt(file_data, delimiter=',')
        x, y = data[:, 1:], data[:, 0]
    else:
        x, y = file_data
    scaler = Normalizer()
    x = scaler.fit_transform(x)
    return x, y

def data_to_hpo(file, hpo=1):
    hpo_x, hpo_y = load_normal_data(file)
    if hpo < 1:
        hpo_x, _, hpo_y, _ = train_test_split(
        hpo_x, hpo_y, test_size=1-hpo, random_state=1, shuffle=True)
    return hpo_x, hpo_y

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
              
def print_report(metric, cm, report_file):
    accl, snl, spl, ppvl, mccl = metric
    with open(report_file, "w") as f:
        tp, fn, fp, tn, sn, sp, acc, mcc, ppv = "tp", "fn", "fp", "tn", "sn", "sp", "acc", "mcc", "ppv"
        line0 = f"   {tp:<4}{fn:<4}{fp:<4}{tn:<4}{sn:<7}{sp:<7}{ppv:<7}{acc:<7}{mcc:<7}\n"
        f.write(line0)
        for idx, line in enumerate(cm):
            (tn, fp), (fn, tp) = line
            acc, sn, sp, ppv, mcc = accl[idx]*100, snl[idx]*100, spl[idx]*100, ppvl[idx]*100, mccl[idx]*100
            linei = f"{idx:<3}{tp:<4}{fn:<4}{fp:<4}{tn:<4}{sn:<7.2f}{sp:<7.2f}{ppv:<7.2f}{acc:<7.2f}{mcc:<7.2f}\n"
            f.write(linei)
        f.write("\n\n")
        f.write(TEXT)
