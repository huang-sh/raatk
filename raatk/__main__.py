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
from pathlib import Path

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
    values = ul.reduce_query(type_id, size) ## 获取氨基酸约化字母表
    for item in values:
        tpi, size, cluster, method = item
        info = f"type{tpi:<3}{size:<3}{cluster:<40}{method}"
        print(info)
    
def sub_reduce(args):
    ul.exist_file(*args.file)
    cluster, type_, size = args.cluster, args.type, args.size
    if (cluster is None) and (type_ is None or size is None):
        print('Missing arguments -t/--type and -s/--size or -c/--cluster')
        return
    for file, out in zip(args.file, args.output):
        if cluster:
            ul.check_aa(cluster)
            aa = cluster.split("-")
            out = Path(out)
            ul.reduce_to_file(file, aa, out)
        elif type_ and size:
            if "-" in type_[0]:
                start, end = type_[0].split("-")
                types = map(str, range(int(start), int(end)+1))
            else:
                types = map(str, type_)
            if "-" in size[0]:
                start, end = size[0].split("-")
                sizes = map(str, range(int(start), int(end)+1))
            else:
                sizes = map(str, size)
            cluster_info = ul.reduce_query(",".join(types), ",".join(sizes))
            out = Path(out)
            out.mkdir(exist_ok=True)
            ul.batch_reduce(file, cluster_info, out)
            if args.naa:
                naa_dir = out / "naa"
                naa_dir.mkdir(exist_ok=True)
                import shutil
                shutil.copyfile(file, naa_dir / "20-ACDEFGHIKLMNPQRSTVWY.txt")
                
def sub_extract(args):
    k, gap, lam, n_jobs = args.kmer, args.gap, args.lam, args.process
    if args.directory:
        out = Path(args.output[0])
        out.mkdir(exist_ok=True)
        ul.batch_extract(args.file, out, k, gap, lam, n_jobs=n_jobs)
    else:
        if args.raa:
            raa = list(args.raa) 
        else:
            raa = list(Path(args.file[0]).stem.split('-')[-1])
        xy_ls = []
        for idx, file in enumerate(args.file):
            feature_file = Path(file)
            xy = ul.extract_feature(feature_file, raa, k, gap, lam)
            if args.index:
                fea_idx = np.genfromtxt(args.index, delimiter='\n').astype(int)
                xy = xy[:, fea_idx]
            if args.label_f:
                y = np.array([[idx]]*xy.shape[0])
                xy = np.hstack([y, xy])
            xy_ls.append(xy)
        if args.merge:
            out = Path(args.output[0])
            seq_mtx = np.vstack(xy_ls)
            ul.write_array(Path(out), seq_mtx)
            exit()
        for idx, o in enumerate(args.output):
            ul.write_array(Path(o), xy_ls[idx])


def sub_hpo(args):
    params = ul.param_grid(args.C, args.gamma)
    x, y = ul.load_data(args.file, normal=True)
    best_c, best_gamma = cp.grid_search(x, y, params)
    print("C: %s, gamma: %s" % (best_c, best_gamma))

def sub_train(args):
    c, g = args.C, args.gamma
    if args.directory:
        in_dir = Path(args.file)
        out_dir = Path(args.output)
        out_dir.mkdir(exist_ok=True)
        cp.batch_train(in_dir, out_dir, c, g, args.process)
    else:
        x, y = ul.load_data(args.file, normal=True)
        model = cp.train(x, y, c, g, probability=True)
        cp.save_model(model, args.output) 

def sub_predict(args):
    x, _ = ul.load_data(args.file, label_exist=False, normal=True)
    model = ul.load_model(args.model)
    y_pred, y_prob = cp.predict(x, model)
    if y_prob is None:
        ul.write_array(args.output, y_pred.reshape(-1,1))
    else:
        ul.write_array(args.output, y_pred.reshape(-1,1), y_prob)
    
def sub_eval(args):
    c, g, cv = args.C, args.gamma, args.cv
    if args.directory:
        out = Path(args.output)
        all_sub_metric_dic = cp.batch_evaluate(Path(args.file), out, cv, c, g, args.process)
        result_json = args.output + ".json"
        ul.save_json(all_sub_metric_dic, result_json)
    else:
        x, y = ul.load_data(args.file, normal=True)
        metric_dic = cp.evaluate(x, y, cv, c, g, probability=True)
        ul.save_report(metric_dic, args.output + '.txt')
        ul.k_roc_curve_plot(metric_dic['y_true'], metric_dic['y_prob'], args.output + '.png')       

def sub_roc(args):
    model = ul.load_model(args.model)
    model.set_params(probability=True)
    x, y = ul.load_data(args.file, normal=True)
    ul.roc_eval(x, y, model, args.output)

def sub_ifs(args):
    ul.exist_file(*args.file)
    C, gamma, step, cv, n_jobs = args.C, args.gamma, args.step, args.cv, args.process
    if args.mix:
        pass
#         x, y = ul.feature_mix(args.file)
#         noraml_x, noraml_y = ul.load_data((x, y))
#         result_ls = cp.feature_select(noraml_x, noraml_y, C, gamma, step, cv, n_jobs)
#         x_tricks = [i for i in range(0, x.shape[1], args.step)]
#         x_tricks.append(x.shape[1])
#         acc_ls = [0] + [i[0][0][0] for i in result_ls]
#         ul.save_y(args.output, x_tricks, acc_ls)
#         max_acc = max(acc_ls)
#         best_n = acc_ls.index(max_acc) * step
#         draw.p_fs(x_tricks, acc_ls, args.output[0]+'.png', max_acc=max_acc, best_n=best_n)
    else:
        for file, out in zip(args.file, args.output): 
            x, y = ul.load_data(file)
            result_ls, sort_idx = cp.feature_select(x, y, C, gamma, step, cv, n_jobs)
            x_tricks = [i for i in range(0, x.shape[1], args.step)]
            x_tricks.append(x.shape[1])
            acc_ls = [0] + [i["OA"] for i in result_ls]
            max_acc = max(acc_ls)
            best_n = acc_ls.index(max_acc) * step
            draw.p_fs(x_tricks, acc_ls, out + '.png', max_acc=max_acc, best_n=best_n)
            best_x = x[:, sort_idx[:best_n]]
            best_file =  out + '_best.csv'
            ul.write_array(best_file, y.reshape(-1, 1), best_x)
            xtricks_arr = np.array(x_tricks[:len(acc_ls)]).reshape(-1, 1)
            acc_arr = np.array(acc_ls).reshape(-1, 1)
            ul.write_array(out+".csv", xtricks_arr, acc_arr)
            ul.write_array(out+f"-{best_n}-idx.csv", sort_idx[:best_n])
            
def sub_plot(args):
    ul.mkdirs(args.outdir)
    with open(args.file, 'r') as f:
        re_dic = json.load(f)
        ul.eval_plot(re_dic, Path(args.outdir), fmt=args.format)

def sub_merge(args):
    mix_data = ul.merge_feature_file(args.label, args.file)
    ul.write_array(args.output, mix_data)

def sub_split(args):
    ts = args.testsize
    if 0 < ts < 1:
        data_train, data_test = ul.split_data(args.file, ts)
        ul.write_array(f"{1-ts}_{args.output}", data_train)
        ul.write_array(f"{ts}_{args.output}", data_test)
    else:
        print("error")

# TODO
def sub_transfer(args):
    pass

class MyArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()
 
def command_parser():
    parser = MyArgumentParser(description='reduce amino acids toolkit', fromfile_prefix_chars='@')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_v = subparsers.add_parser('view', help='view the reduce amino acids scheme')
    parser_v.add_argument('-t', '--type', nargs='+', type=int, required=True, 
                          choices=list([i for i in range(1, 75)]),help='type id')
    parser_v.add_argument('-s', '--size', nargs='+', type=int, required=True, 
                          choices=list([i for i in range(2, 20)]), help='reduce size')
    parser_v.set_defaults(func=sub_view)

    parser_r = subparsers.add_parser('reduce', help='reduce sequence')
    parser_r.add_argument('file', nargs='+', help='fasta file paths')
    parser_r.add_argument('-t', '--type', nargs='+', help='type id')
    parser_r.add_argument('-s', '--size', nargs='+', help='reduce size')
    parser_r.add_argument('-c', '--cluster', help='customized cluster')
    parser_r.add_argument('-naa', action='store_true', help='natural amino acid')
    parser_r.add_argument('-o', '--output', nargs='+', help='output file or directory')
    parser_r.set_defaults(func=sub_reduce)

    parser_ex = subparsers.add_parser('extract', help='extract sequence feature')
    parser_ex.add_argument('file', nargs='+', help='fasta files')
    parser_ex.add_argument('-d', '--directory', action='store_true', help='feature directory')
    parser_ex.add_argument('-k', '--kmer', type=int, choices=[1,2,3], 
                          required=True, help='K-tuple or k-mer value')
    parser_ex.add_argument('-g', '--gap', type=int, default=0, help='gap value')
    parser_ex.add_argument('-l', '--lam', type=int, default=0, 
                          help='lambda-correlation value')
    parser_ex.add_argument('-raa', help='reduced amino acid cluster', default="ACDEFGHIKLMNPQRSTVWY")
    parser_ex.add_argument('-idx', '--index', default=None, help='feature index')
    parser_ex.add_argument('-m', '--merge', action='store_true', help='merge feature files into one')
    parser_ex.add_argument('-o', '--output', nargs='+', required=True, help='output directory')
    parser_ex.add_argument('-p', '--process',type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=1, help='cpu number')
    parser_ex.add_argument('--label-f', action='store_false', help='feature label')
    parser_ex.set_defaults(func=sub_extract)

    parser_hpo = subparsers.add_parser('hpo', help='hpyper-parameter optimization')
    parser_hpo.add_argument('file', help='feature file for hpyper-parameter optimization')
    parser_hpo.add_argument('-c', '--C',  nargs='+', required=True, type=int,
                            help='regularization parameter value range [start, stop, [num]]')
    parser_hpo.add_argument('-g', '--gamma', nargs='+', required=True, type=int,
                            help='Kernel coefficient value range [start, stop, [num]]')
    parser_hpo.set_defaults(func=sub_hpo)
    
    parser_t = subparsers.add_parser('train', help='train model')
    parser_t.add_argument('file', help='feature file to train')
    parser_t.add_argument('-d', '--directory', action='store_true', help='feature directory to train')
    parser_t.add_argument('-o', '--output',required=True, help='output directory')
    parser_t.add_argument('-c', '--C', required=True, type=float, help='regularization parameter')
    parser_t.add_argument('-g', '--gamma', required=True, type=float, help='Kernel coefficient')
    parser_t.add_argument('-p', '--process',type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=1, help='cpu number')
    parser_t.set_defaults(func=sub_train)
    
    parser_p = subparsers.add_parser('predict', help='predict')
    parser_p.add_argument('file', help='feature file to predict')
    parser_p.add_argument('-m', '--model', required=True, help='model to predict')
    parser_p.add_argument('-o', '--output', required=True, help='output directory')
    parser_p.set_defaults(func=sub_predict)

    parser_ev = subparsers.add_parser('eval', help='evaluate models')
    parser_ev.add_argument('file', help='feature file to evaluate')
    parser_ev.add_argument('-d', '--directory', action='store_true', help='feature directory to evaluate')
    parser_ev.add_argument('-o', '--output', required=True, help='output directory')
    parser_ev.add_argument('-cv', type=float, default=-1, help='cross validation fold')
    parser_ev.add_argument('-c', '--C', required=True, type=float, help='regularization parameter')
    parser_ev.add_argument('-g', '--gamma', required=True, type=float, help='Kernel coefficient')
    parser_ev.add_argument('-p', '--process',type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=int(os.cpu_count()/2), help='cpu numbers')
    parser_ev.set_defaults(func=sub_eval)
    
    parser_roc = subparsers.add_parser('roc', help='model roc evaluation')
    parser_roc.add_argument('file', help='feature file for roc')
    parser_roc.add_argument('-m', '--model', required=True, help='model')
    parser_roc.add_argument('-o', '--output', required=True, help='output directory')
    parser_roc.set_defaults(func=sub_roc)

    parser_f = subparsers.add_parser("ifs", help='incremental feature selction using ANOVA')
    parser_f.add_argument('file', nargs='+', help='feature file')
    parser_f.add_argument('-s', '--step', default=10, type=int, help='feature file')
    parser_f.add_argument('-o', '--output', nargs='+', required=True, help='output folder')
    parser_f.add_argument('-c', '--C', required=True, type=float, help='regularization parameter')
    parser_f.add_argument('-g', '--gamma', required=True, type=float, help='Kernel coefficient')
    parser_f.add_argument('-cv', '--cv', type=float, default=-1, help='cross validation fold')
    parser_f.add_argument('-mix', action='store_true', help='feature mix')
    parser_f.add_argument('-p', '--process', type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=int(os.cpu_count()/2), help='cpu core number')
    parser_f.set_defaults(func=sub_ifs)
    
    parser_p = subparsers.add_parser("plot", help='plot evaluate result')
    parser_p.add_argument('file', help='the result json file')
    parser_p.add_argument('-fmt', '--format', default="png", help='figure format')
    parser_p.add_argument('-o', '--outdir', required=True, help='output directory')
    parser_p.set_defaults(func=sub_plot)
       
    parser_m = subparsers.add_parser('merge', help='merge files into one')
    parser_m.add_argument('file', nargs='+', help='file paths')
    parser_m.add_argument('-l', '--label', nargs='+', type=int, help='file format')
    parser_m.add_argument('-o', '--output', help='output file')
    parser_m.set_defaults(func=sub_merge)

    parser_s = subparsers.add_parser('split', help='split file data')
    parser_s.add_argument('file', help='file path')
    parser_s.add_argument('-ts', '--testsize', type=float, help='test size')
    parser_s.add_argument('-o', '--output', help='output file')
    parser_s.set_defaults(func=sub_split)
    
    parser_tra = subparsers.add_parser('transfer', help='transfer file format')
    parser_tra.add_argument('file', help='file path')
    parser_tra.add_argument('-fmt', '--format', choices=["fa", "csv"], help='file format')
    parser_tra.add_argument('-o', '--output', help='output file')
    parser_tra.set_defaults(func=sub_transfer)
    
    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        pass

if __name__ == '__main__':
    command_parser()
