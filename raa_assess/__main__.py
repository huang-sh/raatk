# -*- coding: utf-8 -*-
"""
    :Author: huangsh
    :Date: 19-1-14 下午3:22
    :Description:
        过早的优化是万恶之源!
        coding...
"""
import os
import json
import argparse
import sqlite3

import numpy as np

try:
    from . import draw
    from . import utils as ul
    from . import compute as cp
except ImportError:
    import draw
    import utils as ul
    import compute as cp  


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
        n_fea_dir = os.path.join(args.o, f'{args.o}_{n}n')
        ul.mkdirs(n_fea_dir)
        ul.reduce_seq(args.f, n_fea_dir, n, cluster_info, args.p)

def sub_eval(args):
    for n in args.k:
        n_fea_dir = os.path.join(args.o, f'{args.o}_{n}n')
        ul.mkdirs(n_fea_dir)
        json_path = os.path.join(args.o, f'{n}n_result.json')
        cp.all_eval(n_fea_dir, json_path, n, args.cv, args.hpod, args.p)
        if args.v:
            with open(json_path, 'r') as f:
                re_dic = json.load(f)
            ul.eval_plot(re_dic, n, args.o, fmt=args.fmt)

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
        acc_ls = cp.feature_mix(args.f, args.p, cv=args.cv, hpo=args.hpod)
        filename = f'mix_feature.{args.fmt}'
        fig_path = os.path.join(args.o, filename)
        draw.p_fs(acc_ls, out=fig_path)
    else:
        for file in args.f: 
            acc_ls = cp.feature_select(file, args.p, cv=args.cv, hpod=args.hpod)
            filename = file.split('.')[0].split(os.sep)[-1] + f'.{args.fmt}'
            fig_path = os.path.join(args.o, filename)
            draw.p_fs(acc_ls, out=fig_path)

def sub_own(args):
    ul.mkdirs(args.o)
    c = args.c
    g = args.g
    for n in args.k:
        cluster = args.cluster.split("-")
        feature_file_path = os.path.join(args.o, f"{len(cluster)}_{n}n.csv")
        if c and g:
            metric, cm = cp.own_func(args.f, feature_file_path, cluster, n, args.hpod, **{"c": c, "gamma": g})
        else:
            metric, cm = cp.own_func(args.f, feature_file_path, cluster, n, args.hpod)
        report_file = os.path.join(args.o, f"{n}n_report.txt")
        ul.print_report(metric, cm, report_file)
    
    
def workflow(args):
    if args.r:
        sub_reduce(args)
    if args.c:
        sub_eval(args)
        

def command_parser():
    parser = argparse.ArgumentParser(description='reduce sequence and classify')
    
    parser.add_argument('-f', nargs='+', help='amino acid fasta files, >= 2')
    parser.add_argument('-r', action='store_true',
                                    help='reduce sequence based on reduce type')
    parser.add_argument('-t', nargs='+', help='type id')
    parser.add_argument('-s', nargs='+', help='reduce size')
    parser.add_argument('-k', nargs='+', type=int, choices=[1,2,3])
    parser.add_argument('-c', action='store_true', help='compute')
    parser.add_argument('-cv', type=float, help='cross validation fold')
    parser.add_argument('-hpo', type=str, help='hyper-parameter optimize,')
    parser.add_argument('-hpod', type=float, default=.6, help='hyper-parameter optimize,')
    parser.add_argument('-o', help='output folder name')
    parser.add_argument('-v', action='store_true', help='plot')
    parser.add_argument('-fmt', default="png", help='the format of figures')
    parser.add_argument('-p', type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=os.cpu_count()/2, help='output folder name')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_v = subparsers.add_parser('view', help='view the reduce amino acids scheme')
    parser_v.add_argument('-t', '--type', nargs='+', type=int, choices=list([i for i in range(1, 74)]),help='type id')
    parser_v.add_argument('-s', '--size', nargs='+', type=int, choices=list([i for i in range(2, 20)]), help='reduce size')
    parser_v.set_defaults(func=sub_view)

    parser_a = subparsers.add_parser('reduce', help='reduce sequence and extract feature')
    parser_a.add_argument('-f', nargs='+', help='fasta files')
    parser_a.add_argument('-k', nargs='+', type=int, choices=[1,2,3], help='feature extract method')
    parser_a.add_argument('-t', nargs='+', help='type id')
    parser_a.add_argument('-s', nargs='+', help='reduce size')
    parser_a.add_argument('-o', help='output folder basename')
    parser_a.add_argument('-p', type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=os.cpu_count()/2, help='output folder name')
    parser_a.set_defaults(func=sub_reduce)

    parser_c = subparsers.add_parser('eval', help='evaluate models')
    parser_c.add_argument('-o', help='the value of reduce command -o')
    parser_c.add_argument('-k', nargs='+', type=int, choices=[1,2,3], help='feature extract method')
    parser_c.add_argument('-cv', type=float, help='cross validation fold')
    parser_c.add_argument('-hpo', type=str, help='hyper-parameter optimize method,')
    parser_c.add_argument('-hpod', type=float, default=.6, help='hyper-parameter optimize data size')
    parser_c.add_argument('-v', action='store_true', help='visual')
    parser_c.add_argument('-fmt', default="png", help='the format of figures')
    parser_c.add_argument('-p', type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=int(os.cpu_count()/2), help='output folder name')
    parser_c.set_defaults(func=sub_eval)
    
    parser_d = subparsers.add_parser("plot", help='analyze and plot evaluate result')
    parser_d.add_argument('-f', nargs='+', help='the result json file')
    parser_d.add_argument('-fmt', default="png", help='the format of figures')
    parser_d.add_argument('-o', help='output folder')
    parser_d.set_defaults(func=sub_plot)

    parser_e = subparsers.add_parser("fs", help='analyze and plot evaluate result')
    parser_e.add_argument('-f', nargs='+', help='feature file')
    parser_e.add_argument('-o', help='output folder')
    parser_e.add_argument('-cv', type=float, default=-1, help='cross validation fold')
    parser_e.add_argument('-hpo', type=str, help='hyper-parameter optimize method')
    parser_e.add_argument('-hpod', type=float, default=.6, help='hyper-parameter optimize data')
    parser_e.add_argument('-fmt', default="png", help='the format of figures')
    parser_e.add_argument('-p', type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=os.cpu_count()/2, help='output folder name')
    parser_e.add_argument('-mix', action='store_true', help='feature mix')
    parser_e.set_defaults(func=sub_fs)
    
    parser_f = subparsers.add_parser("own", help='use your own raa')
    parser_f.add_argument('-f', nargs='+', help='fasta files')
    parser_f.add_argument('-cluster', help='amino acid reduction scheme')
    parser_f.add_argument('-k', nargs='+', type=int, choices=[1,2,3], help='feature extract method')
    parser_f.add_argument('-o', help='output folder')
    parser_f.add_argument('-cv', type=float, help='cross validation fold')
    parser_f.add_argument('-hpo', type=str, help='hyper-parameter optimize method')
    parser_f.add_argument('-hpod', type=float, default=.6, help='hyper-parameter optimize data')
    parser_f.add_argument('-c', type=float, default=0, help='svm parameter C')
    parser_f.add_argument('-g', type=float, default=0, help='svm parameter gamma')   
    parser_f.set_defaults(func=sub_own) 
    
    args = parser.parse_args()
    
    try:
        args.func(args)
    except AttributeError:
        workflow(args)

if __name__ == '__main__':
    command_parser()

