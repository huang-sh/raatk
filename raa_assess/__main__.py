# -*- coding: utf-8 -*-
"""
    :Author: huangsh
    :Date: 19-1-14 下午3:22
    :Description:
        过早的优化是万恶之源!
"""
import os
import json
import argparse
import sqlite3

import numpy as np

from . import draw
from . import utils as ul
from . import compute as cp


def sub_view(args):
    type_id = ','.join(map(str, args.type))
    size = ','.join(map(str, args.size))
    values = ul.reduce_query(type_id, size)
    for item in values:
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
    cluster_info = ul.reduce_query(type_id, size)
    for n in args.k:
        ul.reduce_seq(args.f, f'{args.o}_{n}n', n, cluster_info, args.p)

def sub_eval(args):
    ul.mkdirs(args.input)
    for n in args.k:
        folder_name = f'{args.input}_{n}n'
        json_path = os.path.join(args.input, f'{n}n_result.json')
        cp.all_eval(folder_name, json_path, n, args.cv, args.hpo, args.p)
        if args.v:
            with open(json_path, 'r') as f:
                re_dic = json.load(f)
            ul.eval_plot(re_dic, n, args.input, fmt=args.fmt)

def sub_plot(args):
    ul.mkdirs(args.o)
    for re_file in args.f:
        with open(re_file, 'r') as f:
            re_dic = json.load(f)
        n = os.path.basename(re_file).split("_")[0][0]
        ul.eval_plot(re_dic, int(n), args.o, fmt=args.fmt)

def sub_fs(args):
    ul.mkdirs(args.o)
    if args.mix:
        acc_ls = cp.feature_mix(args.f, cv=args.cv, hpo=args.hpo)
        filename = f'mix_feature.{args.fmt}'
        fig_path = os.path.join(args.o, filename)
        draw.p_fs(acc_ls, out=fig_path)
    else:
        for file in args.f: 
            acc_ls = cp.feature_select(file, cv=args.cv, hpo=args.hpo)
            filename = file.split('.')[0].split(os.sep)[-1] + f'.{args.fmt}'
            fig_path = os.path.join(args.o, filename)
            draw.p_fs(acc_ls, out=fig_path)

def sub_own(args):
    ul.mkdirs(args.o)
    for n in args.k:
        cluster = args.cluster.split("-")
        feature_file_path = os.path.join(args.o, f"{len(cluster)}_{n}n.csv")
        metric, cm = cp.own_func(args.f, feature_file_path, cluster, n)
        report_file = os.path.join(args.o, f"{n}n_report.txt")
        ul.print_report(metric, cm, report_file)
    
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
    parser_a.add_argument('-f', nargs='+', help='fasta files')
    parser_a.add_argument('-k', nargs='+', type=int, choices=[1,2,3], help='feature extract method')
    parser_a.add_argument('-t', nargs='+', help='type id')
    parser_a.add_argument('-s', nargs='+', help='reduce size')
    parser_a.add_argument('-o', help='output folder name')
    parser_a.add_argument('-p', type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=os.cpu_count()/2, help='output folder name')
    parser_a.set_defaults(func=sub_reduce)

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
    parser_d.add_argument('-f', nargs='+', help='the result json file')
    parser_d.add_argument('-fmt', default="png", help='the format of figures')
    parser_d.add_argument('-o', help='output folder')
    parser_d.set_defaults(func=sub_plot)

    parser_e = subparsers.add_parser("fs", help='analyze and plot evaluate result')
    parser_e.add_argument('-f', nargs='+', help='feature file')
    parser_e.add_argument('-o', help='output folder')
    parser_e.add_argument('-cv', type=float, help='cross validation fold')
    parser_e.add_argument('-hpo', type=float, help='hyper-parameter optimize,')
    parser_e.add_argument('-fmt', default="png", help='the format of figures')
    parser_e.add_argument('-mix', action='store_true', help='feature mix')
    parser_e.set_defaults(func=sub_fs)
    
    parser_f = subparsers.add_parser("own", help='use your own raa')
    parser_f.add_argument('-f', nargs='+', help='fasta files')
    parser_f.add_argument('-cluster', help='fasta files')
    parser_f.add_argument('-k', nargs='+', type=int, choices=[1,2,3], help='feature extract method')
    parser_f.add_argument('-o', help='output folder')
    parser_f.add_argument('-cv', type=float, help='cross validation fold')
    parser_f.add_argument('-hpo', type=float, help='hyper-parameter optimize,')
    parser_f.set_defaults(func=sub_own) 
    
    args = parser.parse_args()
    
    try:
        args.func(args)
    except AttributeError:
        pass

if __name__ == '__main__':
    command_parser()


