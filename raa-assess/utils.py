import os
import csv
import zipfile
import sqlite3
from concurrent import futures

import numpy as np
import pysnooper

import feature

MAX_WORKERS = 676
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
            base_aac = feature.aac(simple_seq, raa, n)
            for a in base_aac:
                line0 = [v for v in a[1]]
                line1 = [idx] + line0
                h.writerow(line1)
                # new_handle.write('\n')
            f.close()

def thread_func(file_list, folder_n, n, clusters):
    to_do_map = {}
    with futures.ThreadPoolExecutor(28) as tpe:
        for idx, item in enumerate(clusters):
            tpi, size, cluster, _ = item
            aa = cluster.split('-')
            aa = [i for i in aa if i]
            try:
                type_path = os.path.join(folder_n, f"type{tpi}")
                os.mkdir(type_path)
            except FileExistsError:
                pass
            # file_list, file_path, aa, n, idx=None,raa=None
            file_path = os.path.join(folder_n, f"type{tpi}", f"{size}_{n}n.csv")
            future = tpe.submit(one_file, file_list, file_path, aa, n, idx=size)
            to_do_map[future] = tpi, size, cluster
        done_iter = futures.as_completed(to_do_map)
        for i in done_iter:
            print(i)

def reduce_seq(file_list, folder, n, cluster_info, p):
    try:
        folder_n = folder
        os.mkdir(folder_n)
    except FileExistsError:
        pass
    to_do_map = {}
    cluster_per = []
    counts = len(cluster_info)
    max_work = min(p, os.cpu_count(), counts)
    max_work = max(1, max_work)
    per = int(counts / max_work)
    with futures.ProcessPoolExecutor(max_work) as ppe:
        for idx, item in enumerate(cluster_info, 1):
            if idx % per == 0:
                clusters = cluster_per.copy()
                future = ppe.submit(thread_func, file_list, folder_n, n, clusters)
                to_do_map[future] = [idx-per, idx]
                cluster_per.clear()
            cluster_per.append(item)
        else:
            future = ppe.submit(thread_func, file_list, folder_n, n, cluster_per)
            to_do_map[future] = [idx-per, idx]
            # cluster_per.clear()
        naa_path = os.path.join(folder_n, f'20_{n}n.csv')
        future = ppe.submit(one_file, file_list, naa_path, NAA, n)
        to_do_map[future] = "20s"
        done_iter = futures.as_completed(to_do_map)
        for f in done_iter:
            idx = to_do_map[f]
            print(f'{n}n --> {idx}', 'has done!')

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


def dic2array(result_dic, key='acc', filter_num=0, cls=0):
    acc_ls = []  # all type acc
    filtered_type_acc = []
    type_ls = [type_id for type_id in result_dic.keys()]
    type_ls.sort(key=lambda x: int(x[4:]))
    all_score_array = np.zeros([len(type_ls), 19])

    filtered_type_ls = type_ls.copy()
    filtered_score_ls = []

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


if __name__ == '__main__':
    pass
