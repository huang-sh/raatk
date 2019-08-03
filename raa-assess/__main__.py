# -*- coding: utf-8 -*-
"""
    :Author: huangsh
    :Date: 19-1-14 下午3:22
    :Description:
        过早的优化是万恶之源!
"""
import os
import sys
import json
import argparse
import cProfile
import sqlite3

import numpy as np
import matplotlib.pyplot as plt
import pysnooper

import feature, draw
import utils as ul
import computation as cp


def eval_plot(result_dic, n, out, fmt='tiff', filter_num = 8):
    key = 'acc'
    all_score, filter_score = ul.dic2array(result_dic, key=key, filter_num=filter_num)
    scores, types = all_score
    f_scores, f_types = filter_score
    f_heatmap_path = os.path.join(out, f'{key}_f{filter_num}-heatmap_{n}n.{fmt}')
    heatmap_path = os.path.join(out, f'{key}_heatmap_{n}n.{fmt}')
    draw.p_acc_heat(f_scores.T, 0.6, 1, f_types, f_heatmap_path)
    draw.p_acc_heat(scores.T, 0.6, 1, types, heatmap_path)

    f_scores_arr = f_scores[f_scores > 0]
    size_arr = np.array([np.arange(2, 21)] * f_scores.shape[0])[f_scores > 0]
    path = os.path.join(out, f'acc_size_density-{n}n.pdf')
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
    com_result = cp.al_comparison(max_acc_fea_file)
    # roc_path = os.path.join(out, f'{n}n_al_roc.{fmt}')
    # draw.p_roc_al(param, roc_path)
    return f_scores_arr, max_acc_fea_file

def sub_view(args):
    type_id = ','.join(map(str, args.type))
    size = ','.join(map(str, args.size))
    values = ul.query(type_id, size)
    for item in values:
        # print(*item)
        tpi, size, cluster, method = item
        info = f"type{tpi:<3}{size:<3}{cluster:<40}{method}"
        print(info)
    
def sub_reduce(args):
    tpi, size = args.t, args.s
    if "-" in tpi[0]:
        ss = tpi[0].split("-")
        tpi = [i for i in range(int(ss[0]), int(ss[1])+1)]
    if "-" in size[0]:
        ss = size[0].split("-")
        size = [i for i in range(int(ss[0]), int(ss[1])+1)]
    type_id = ','.join(map(str, tpi))
    size = ','.join(map(str, size))
    cluster_info = ul.query(type_id, size)
    for n in args.k:
        # ul.all_type(args.f, f'{args.o}_{n}n', n)
        ul.reduce_seq(args.f, f'{args.o}_{n}n', n, cluster_info, args.p)

def sub_eval(args):
    try:
        os.mkdir(args.input)
    except FileExistsError:
        pass
    for n in args.k:
        folder_name = f'{args.input}_{n}n'
        json_path = os.path.join(args.input, f'{n}n_result.json')
        cp.all_eval(folder_name, json_path, n, args.cv, args.hpo)
        if args.v:
            with open(json_path, 'r') as f:
                re_dic = json.load(f)
            eval_plot(re_dic, n, args.input, fmt=args.fmt)

def sub_plot(args):
    try:
        os.mkdir(args.out)
    except FileExistsError:
        pass
    for re_file in args.file:
        with open(re_file, 'r') as f:
            re_dic = json.load(f)
        n = os.path.basename(re_file).split("_")[0][0]
        eval_plot(re_dic, int(n), args.out, fmt=args.fmt)

def sub_fs(args):
     for file in args.f:
        acc_ls = cp.feature_select(file)
        fig_path = os.path.join(args.o, file.split('.')[0])
        draw.p_fs(acc_ls, out=fig_path)
#
# def workflow(args):
#     file_dic = {}
#     fs_file = []
#     diff_n_acc = []
#     for n in args.k:
#         folder = f'{folder}_{n}n'
#         json_path = os.path.join(args.o, f'{n}n_result.json')
#         file_dic[n] = json_path
#         if args.r:
#             ul.all_type(args.f, folder, n)
#         if args.te:
#             cp.all_drill(folder_name)
#             cp.all_eval(folder_name, json_path, args.cv)
#         if os.path.exists(json_path):
#             with open(json_path, 'r') as f:
#                 result_dic = json.load(f)
#             f_scores_arr, max_acc_fea_file = plot_n_eval(result_dic, n, args.o, args.o)
#             fs_file.append(max_acc_fea_file)
#             diff_n_acc.append(fs_file)
#     density_path = os.path.join(args.o, f'acc_density_{n}.pdf')
#     draw.p_univariate_density(diff_n_acc, [f'{i}n' for i in args.k], density_path)
#     for idx, file in enumerate(fs_file):
#         acc_ls = cp.feature_select(file)
#         fig_path = os.path.join(args.o, file.split('.')[0])
#         draw.p_fs(acc_ls, out=fig_path)

def command_parser():
    parser = argparse.ArgumentParser(description='reduce sequence and classify')
    parser.add_argument('-r', '--reduce', action='store_true',
                                    help='reduce sequence based on reduce type')
    parser.add_argument('-f', '--file', nargs='+', help='input file')
    parser.add_argument('-k', nargs='+', type=int, choices=[1,2,3])
    parser.add_argument('-o', '--output', help='output folder name')
    parser.add_argument('-c', '--compute', action='store_true', help='compute')
    parser.add_argument('-p', '--plot', action='store_true', help='plot')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_v = subparsers.add_parser('view', help='view the reduce amino acids scheme')
    parser_v.add_argument('--type', nargs='+', type=int, choices=list([i for i in range(1, 74)]),help='type id')
    parser_v.add_argument('--size', nargs='+', type=int, choices=list([i for i in range(2, 20)]), help='reduce size')
    parser_v.set_defaults(func=sub_view)

    parser_a = subparsers.add_parser('reduce', help='reduce sequence and extract feature')
    parser_a.add_argument('-f', nargs='+', help='reduce files')
    parser_a.add_argument('-k', nargs='+', type=int, choices=[1,2,3], help='feature extract method')
    parser_a.add_argument('-t', nargs='+', help='type id')
    parser_a.add_argument('-s', nargs='+', help='reduce size')
    parser_a.add_argument('-o', help='output folder name')
    parser_a.add_argument('-p', type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=os.cpu_count()/2, help='output folder name')
    parser_a.set_defaults(func=sub_reduce)

    # parser_b = subparsers.add_parser('train', help='train model with reduced data')
    # parser_b.add_argument('-input', help='feature folder')
    # parser_b.add_argument('-k', nargs='+', type=int, choices=[1,2,3], help='feature extract method')
    # parser_b.add_argument('-hp', help='parameter search')
    # parser_b.set_defaults(func=sub_train)

    parser_c = subparsers.add_parser('eval', help='evaluate models')
    parser_c.add_argument('-input', help='feature folder')
    parser_c.add_argument('-k', nargs='+', type=int, choices=[1,2,3], help='feature extract method')
    parser_c.add_argument('-cv', type=float, help='cross validation fold')
    parser_c.add_argument('-hpo', type=float, help='hyper-parameter optimize,')
    parser_c.add_argument('-v', action='store_true', help='if visual')
    parser_c.add_argument('-fmt', default="png", help='the format of figures')
    parser_c.add_argument('-p', type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=os.cpu_count()/2, help='output folder name')
    parser_c.set_defaults(func=sub_eval)
    
    parser_d = subparsers.add_parser("plot", help='analyze and plot evaluate result')
    parser_d.add_argument('-file', nargs='+', help='the result json file')
    parser_d.add_argument('-fmt', default="png", help='the format of figures')
    parser_d.add_argument('-out', help='output folder')
    parser_d.set_defaults(func=sub_plot)

    parser_e = subparsers.add_parser("fs", help='analyze and plot evaluate result')
    parser_e.add_argument('-f', nargs='+', help='the result json file')
    parser_e.add_argument('-fmt', default="png", help='the format of figures')
    parser_e.set_defaults(func=sub_fs)
    
    args = parser.parse_args()
    
    try:
        args.func(args)
        # cProfile.run("args.func(args)")
    except AttributeError:
        pass


if __name__ == '__main__':
    command_parser()


