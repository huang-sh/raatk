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

def parse_view(args, sub_parser):
    parser_v = sub_parser.add_parser('view', add_help=False, prog='raatk view',
                         usage='%(prog)s -t TYPE {1,2,3,...,73,74} -s SIZE {2,3,...,18,19}')
    parser_v.add_argument('-h', '--help', action='help')
    parser_v.add_argument('-t', '--type', nargs='+', type=int, required=True, 
                          choices=list([i for i in range(1, 75)]), help='type id')
    parser_v.add_argument('-s', '--size', nargs='+', type=int, required=True, 
                          choices=list([i for i in range(2, 20)]), help='reduce size')
    parser_v.set_defaults(func=sub_view)
    view_args = parser_v.parse_args(args)
    view_args.func(view_args)

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
              
def parse_reduce(args, sub_parser):
    parser_r = sub_parser.add_parser('reduce', add_help=False, prog='raatk reduce')
    parser_r.add_argument('-h', '--help', action='help')
    parser_r.add_argument('file', nargs='+', help='fasta file paths')
    parser_r.add_argument('-t', '--type', nargs='+', help='type id')
    parser_r.add_argument('-s', '--size', nargs='+', help='reduce size')
    parser_r.add_argument('-c', '--cluster', help='customized cluster')
    parser_r.add_argument('-naa', action='store_true', help='natural amino acid')
    parser_r.add_argument('-o', '--output', nargs='+', help='output file or directory')
    parser_r.set_defaults(func=sub_reduce)
    reduce_args = parser_r.parse_args(args)
    reduce_args.func(reduce_args)

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

def parse_extract(args, sub_parser):
    parser_ex = sub_parser.add_parser('extract', add_help=False, prog='raatk extract')
    parser_ex.add_argument('-h', '--help', action='help')
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
    extract_args = parser_ex.parse_args(args)
    extract_args.func(extract_args)

def sub_hpo(args):
    pass
    # a(**dict(args._get_kwargs()))
    # params = ul.param_grid(args.C, args.gamma)
    # x, y = ul.load_data(args.file, normal=True)
    # best_c, best_gamma = cp.grid_search(x, y, params)
    # print("C: %s, gamma: %s" % (best_c, best_gamma))

#TODO
def parse_hpo(args, sub_parser):
    pass
    # parser = sub_parser.add_parser('hpo', add_help=False, prog='raatk hpo')
    # parser.add_argument('-h', '--help', action='help')
    # parser.add_argument('file', help='feature file for hpyper-parameter optimization')
    # parser.add_argument('-clf', '--clf', default='svm', choices=['svm', 'rbf', 'knn'],
    #                                 help='classifier selection')
    # parser.add_argument('-jobs','--n_jobs', type=int, help='the number of parallel jobs to run')

    # knn = parser.add_argument_group('KNN', description='K-nearest neighbors classifier')
    # knn.add_argument('--n_neighbors', type=int, nargs='+',
    #                     help='number of neighbors [start stop step]')
    # knn.add_argument('--weights', choices=['uniform', 'distance'], default=['uniform'], 
    #                     nargs='+', help='weight function used in prediction') 
    # knn.add_argument('--algorithm', choices=['auto', 'ball_tree', 'kd_tree', 'brute'], nargs='+',
    #                     default=['auto'], help='algorithm used to compute the nearest neighbors')
    # knn.add_argument('--leaf_size', type=int, default=30,
    #                     help='leaf size passed to BallTree or KDTree')

    # svm = parser.add_argument_group('SVM', description='Parameters for SVM classifier')
    # svm.add_argument('-c', '--C-range',  nargs='+', required=True, type=int,
    #                         help='regularization parameter value range [start, stop, [num]]')
    # svm.add_argument('-g', '--gamma-range', nargs='+', required=True, type=int,
    #                         help='Kernel coefficient value range [start, stop, [num]]')
    # svm.add_argument('--kernel', nargs='+', required=True, choices=['rbf', 'linear'],
    #                         help='kernel function')
    
    # parser.set_defaults(func=sub_hpo)
    # hpo_args = parser.parse_args(args)
    # hpo_args.func(hpo_args)

def clf_parser(parser):
    svm = parser.add_argument_group('SVM', description='Parameters for SVM classifier')
    svm.add_argument('-c', '--C', default=1, type=float, help='regularization parameter')
    svm.add_argument('-g', '--gamma', default='scale', help='kernel coefficient')
    svm.add_argument('-prob', '--probability', action='store_true', help='kernel coefficient')
    svm.add_argument('-k', '--kernel', choices=['rbf', 'linear'], default='rbf',
                            help='specifies the kernel type to be used in the algorithm ')  
    svm.add_argument('-dfs', '--decision_function_shape', default='ovo', choices=['ovr', 'ovo'], 
                            help='decision function shape')
    svm.add_argument('--class_weight', default='balanced', choices=['balanced'],
                                help='default: balanced')
    svm.add_argument('-rs', '--random_state', type=int, default=1, help='random state')
    knn = parser.add_argument_group('KNN', description='K-nearest neighbors classifier')
    knn.add_argument('-n', '--n_neighbors', type=int, default=5,
                        help='number of neighbors to use by default for kneighbors queries')
    knn.add_argument('-w', '--weights', choices=['uniform', 'distance'], default='uniform', 
                        help='weight function used in prediction') 
    knn.add_argument('-al', '--algorithm', choices=['auto', 'ball_tree', 'kd_tree', 'brute'], 
                        default='auto', help='algorithm used to compute the nearest neighbors')
    knn.add_argument('-lfs', '--leaf_size', type=int, default=30,
                        help='leaf size passed to BallTree or KDTree')
    knn.add_argument('-jobs','--n_jobs', type=int, default=1, 
                            help='the number of parallel jobs to run for neighbors search')
    rf = parser.add_argument_group('RF', description='Parameters for random forest classifier') 
    rf.add_argument('-trees', '--n_estimators', type=int, default=100,
                        help='the number of trees in the forest')
    rf.add_argument('-features', '--max_features', default='auto',
                        help='the number of features to consider when looking for the best split') 
    knn.add_argument('-jobs','--n_jobs', type=int, default=1, 
                            help='the number of jobs to run in parallel')
    rf.add_argument('-rs', '--random_state', type=int, default=1, help='random state')
    return parser

def sub_train(args):
    clf = ul.select_clf(args)
    x, y = ul.load_data(args.file, normal=True)
    cp.train(x, y, clf, args.output)

def parse_train(args, sub_parser):
    parser = sub_parser.add_parser('train', add_help=False, prog='raatk train',
                                     conflict_handler='resolve')
    parser.add_argument('-h', '--help', action='help')
    parser.add_argument('file', help='feature file to train')
    parser.add_argument('-clf', '--clf', default='svm', choices=['svm', 'rbf', 'knn'],
                                help='classifier selection')
    parser.add_argument('-o', '--output',required=True, help='output directory')
    parser.add_argument('-jobs','--n_jobs', type=int, default=1, 
                            help='the number of parallel jobs to run')
    clf_parser(parser)
    parser.set_defaults(func=sub_train)
    train_args = parser.parse_args(args)
    train_args.func(train_args)

def sub_predict(args):
    x, _ = ul.load_data(args.file, label_exist=False, normal=True)
    model = ul.load_model(args.model)
    y_pred, y_prob = cp.predict(x, model)
    if y_prob is None:
        ul.write_array(args.output, y_pred.reshape(-1,1))
    else:
        ul.write_array(args.output, y_pred.reshape(-1,1), y_prob)

def parse_predict(args, sub_parser):
    parser = sub_parser.add_parser('predict', add_help=False, prog='raatk predict')
    parser.add_argument('-h', '--help', action='help')
    parser.add_argument('file', help='feature file to predict')
    parser.add_argument('-m', '--model', required=True, help='model to predict')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.set_defaults(func=sub_predict)
    predict_args = parser.parse_args(args)
    predict_args.func(predict_args)

def sub_eval(args):
    clf = ul.select_clf(args)
    if args.directory:
        out = Path(args.output)
        all_metric_dic = cp.batch_evaluate(Path(args.file), out,args.cv, clf, args.process)
        result_json = args.output + ".json"
        ul.metric_dict2json(all_metric_dic, result_json)
    else:
        x, y = ul.load_data(args.file, normal=True)
        metric_dic = cp.evaluate(x, y, args.cv, clf)
        ul.save_report(metric_dic, args.output + '.txt')
        # ul.k_roc_curve_plot(metric_dic['y_true'], metric_dic['y_prob'], args.output + '.png')       

def parse_eval(args, sub_parser):
    parser = sub_parser.add_parser('eval', add_help=False, prog='raatk eval',
                                        conflict_handler='resolve')
    parser.add_argument('-h', '--help', action='help')
    parser.add_argument('file', help='feature file to evaluate')
    parser.add_argument('-d', '--directory', action='store_true', help='feature directory to evaluate')
    parser.add_argument('-clf', '--clf', choices=['svm', 'rf', 'knn'], default='svm', 
                                help='classifier selection')
    parser.add_argument('-cv', type=int, default=5, help='cross validation fold')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('-p', '--process',type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=1, help='cpu numbers')     
    clf_parser(parser)
    parser.set_defaults(func=sub_eval)
    eval_args = parser.parse_args(args)
    eval_args.func(eval_args)

def sub_roc(args):
    x, y = ul.load_data(args.file, normal=True)   
    if args.model:
        cv = None
        clf = ul.load_model(args.model)
    else:
        cv = args.cv
        clf = ul.select_clf(args)
    ul.roc_auc_save(clf, x, y, cv, args.format, args.output)

def parse_roc(args, sub_parser):
    parser = sub_parser.add_parser('roc', add_help=False, prog='raatk roc',
                                        conflict_handler='resolve')
    parser.add_argument('-h', '--help', action='help')
    parser.add_argument('file', help='feature file for roc')
    parser.add_argument('-cv', type=int, default=5, help='cross validation fold')
    me_group = parser.add_mutually_exclusive_group()
    me_group.add_argument('-m', '--model', help='model')
    me_group.add_argument('-clf', '--clf', choices=['svm', 'rf', 'knn'], 
                                help='classifier selection')
    fmt_choices = ['eps', 'pdf', 'png', 'ps', 'raw', 'rgba', 'svg', 'txt']
    parser.add_argument('-fmt', '--format', default="png",
                            choices=fmt_choices, help='figure format')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    clf_parser(parser)
    parser.set_defaults(func=sub_roc)
    roc_args = parser.parse_args(args)
    roc_args.func(roc_args)

def sub_ifs(args):
    ul.exist_file(*args.file)
    step, cv, n_jobs = args.step, args.cv, args.process
    clf = ul.select_clf(args)
    if args.mix:
        pass
    else:
        for file, out in zip(args.file, args.output): 
            x, y = ul.load_data(file)
            result_ls, sort_idx = cp.feature_select(x, y, step, cv, clf, n_jobs)
            x_tricks = [i for i in range(0, x.shape[1], step)]
            x_tricks.append(x.shape[1])
            mean_mt = ul.mean_metric
            acc_ls = [0] + [mean_mt(i)['acc'] for i in result_ls]
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
            
def parse_ifs(args, sub_parser):
    parser = sub_parser.add_parser('ifs', add_help=False, prog='raatk ifs',
                                        conflict_handler='resolve')
    parser.add_argument('-h', '--help', action='help')
    parser.add_argument('file', nargs='+', help='feature file')
    parser.add_argument('-s', '--step', default=10, type=int, help='feature file')
    parser.add_argument('-clf', '--clf', default='svm', choices=['svm', 'rf', 'knn'],
                             help='classifier selection')
    parser.add_argument('-cv', '--cv', type=int, default=5, help='cross validation fold')                             
    parser.add_argument('-o', '--output', nargs='+', required=True, help='output folder')
    parser.add_argument('-mix', action='store_true', help='feature mix')
    parser.add_argument('-p', '--process', type=int, choices=range(1, os.cpu_count()),
                                 default=1, help='cpu core number')
    clf_parser(parser)
    parser.set_defaults(func=sub_ifs)
    ifs_args = parser.parse_args(args)
    ifs_args.func(ifs_args)

def sub_plot(args):
    ul.mkdirs(args.outdir)
    with open(args.file, 'r') as f:
        re_dic = json.load(f)
        ul.eval_plot(re_dic, Path(args.outdir), fmt=args.format)

def parse_plot(args, sub_parser):
    parser = sub_parser.add_parser('plot', add_help=False, prog='raatk plot')
    parser.add_argument('-h', '--help', action='help')
    parser.add_argument('file', help='the result json file')
    fmt_choices = ['eps', 'pdf', 'png', 'ps', 'raw', 'rgba', 'svg']
    parser.add_argument('-fmt', '--format', default="png",
                            choices=fmt_choices, help='figure format')
    # parser.add_argument('-dpi', default=1000, type=int, help='figure format')
    parser.add_argument('-o', '--outdir', required=True, help='output directory')
    parser.set_defaults(func=sub_plot)
    plot_args = parser.parse_args(args)
    plot_args.func(plot_args)

def sub_merge(args):
    mix_data = ul.merge_feature_file(args.label, args.file)
    ul.write_array(args.output, mix_data)

def parse_merge(args, sub_parser):
    parser = sub_parser.add_parser('merge', add_help=False, prog='raatk merge')
    parser.add_argument('-h', '--help', action='help')
    parser.add_argument('file', nargs='+', help='file paths')
    parser.add_argument('-l', '--label', nargs='+', type=int, help='file format')
    parser.add_argument('-o', '--output', help='output file')
    parser.set_defaults(func=sub_merge)
    merge_args = parser.parse_args(args)
    merge_args.func(merge_args)

def sub_split(args):
    ts = args.testsize
    if 0 < ts < 1:
        data_train, data_test = ul.split_data(args.file, ts)
        ul.write_array(f"{1-ts}_{args.output}", data_train)
        ul.write_array(f"{ts}_{args.output}", data_test)
    else:
        print("error")

def parse_split(args, sub_parser):
    parser = sub_parser.add_parser('split', add_help=False, prog='raatk split')
    parser.add_argument('-h', '--help', action='help')
    parser.add_argument('file', help='file path')
    parser.add_argument('-ts', '--testsize', type=float, help='test size')
    parser.add_argument('-o', '--output', help='output file')
    parser.set_defaults(func=sub_split)
    split_args = parser.parse_args(args)
    split_args.func(split_args)

# TODO
def sub_transfer(args):
    pass

def parse_transfer(args, sub_parser):
    parser = sub_parser.add_parser('transfer', add_help=False, prog='raatk transfer')
    parser.add_argument('-h', '--help', action='help')
    parser.set_defaults(func=sub_transfer)
    transfer_args = parser.parse_args(args)
    transfer_args.func(transfer_args)


class MyArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()
 

def command_parser():
    parser = MyArgumentParser(description='reduce amino acids toolkit', fromfile_prefix_chars='@', conflict_handler='resolve')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_v = subparsers.add_parser('view', add_help=False, help='view reduced amino acids alphabet')
    parser_v.set_defaults(func=parse_view)
    parser_r = subparsers.add_parser('reduce', add_help=False, help='reduce amino acid sequence')
    parser_r.set_defaults(func=parse_reduce)
    parser_ex = subparsers.add_parser('extract', add_help=False, help='extract sequence feature')
    parser_ex.set_defaults(func=parse_extract)
    parser_hpo = subparsers.add_parser('hpo', add_help=False, help='hpyper-parameter optimization')
    parser_hpo.set_defaults(func=parse_hpo)
    parser_ev = subparsers.add_parser('eval', add_help=False, help='evaluate model')
    parser_ev.set_defaults(func=parse_eval)    
    parser_t = subparsers.add_parser('train', add_help=False, help='train model')
    parser_t.set_defaults(func=parse_train)
    parser_p = subparsers.add_parser('predict', add_help=False, help='predict data')
    parser_p.set_defaults(func=parse_predict)
    parser_roc = subparsers.add_parser('roc', add_help=False, help='roc curve evaluation')
    parser_roc.set_defaults(func=parse_roc)
    parser_f = subparsers.add_parser("ifs", add_help=False, help='incremental feature selction using ANOVA')
    parser_f.set_defaults(func=parse_ifs)
    parser_p = subparsers.add_parser("plot", add_help=False, help='visualization of evaluation result')
    parser_p.set_defaults(func=parse_plot) 
    parser_m = subparsers.add_parser('merge', add_help=False, help='merge files into one')
    parser_m.set_defaults(func=parse_merge)
    parser_s = subparsers.add_parser('split', add_help=False, help='split file data')
    parser_s.set_defaults(func=parse_split)
    parser_tra = subparsers.add_parser('transfer', add_help=False, help='transfer file format')
    parser_tra.set_defaults(func=parse_transfer)

    tmp, kown_args = parser.parse_known_args()
    try:
        tmp.func(kown_args, subparsers)
    except AttributeError:
        pass

if __name__ == '__main__':
    command_parser()
