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
    ul.exist_file(*args.file)
    cluster, type_, size = args.cluster, args.type, args.size
    if (cluster is None) and (type_ is None or size is None):
        print('Missing arguments -t/--type and -s/--size or -c/--cluster')
        return
    for file, out in zip(args.file, args.output):
        if cluster:
            ul.check_aa(cluster)
            aa = cluster.split("-")
            out = Path(out + "-" + "".join([i[0] for i in aa]))
            ul.reduce_to_file(file, aa, out)
        elif type_ and size:
            if "-" in type_[0]:
                ss = type_[0].split("-")
                types = [str(i) for i in range(int(ss[0]), int(ss[1])+1)]
            if "-" in size[0]:
                ss = size[0].split("-")
                sizes = [str(i) for i in range(int(ss[0]), int(ss[1])+1)]
            cluster_info = ul.reduce_query(",".join(types), ",".join(sizes))
            out = Path(out)
            out.mkdir(exist_ok=True)
            ul.batch_reduce(file, cluster_info, out)
            if args.naa:
                naa_dir = out / "naa"
                naa_dir.mkdir()
                import shutil
                shutil.copyfile(file, naa_dir / "20-ACDEFGHIKLMNPQRSTVWY")
                
def sub_extract(args):
    k, gap, lam, n_jobs = args.kmer, args.gap, args.lam, args.process
    if args.directory:
        if args.merge:
            out = Path(args.output[0])
            if out.exists():
                raise ValueError(f"{out} has existed")
            for idx, dire in enumerate(args.file):
                idx = idx if args.label_f else None
                indir = Path(dire)
                out.mkdir(exist_ok=True)
                ul.batch_extract(indir, out, k, gap, lam, label=idx,
                                  mode="a+", n_jobs=n_jobs)
        else:
            for idx, (dire, out) in enumerate(zip(args.file, args.output)):
                label = idx if args.label_f else None
                indir, outdir = Path(dire), Path(out)
                outdir.mkdir(exist_ok=True)
                ul.batch_extract(indir, outdir, k, gap, lam, label=label,
                                  mode="w", n_jobs=n_jobs)
    else:
        if args.merge: 
            out = Path(args.output[0])
            if out.exists():
                raise ValueError(f"{out} has existed")
            for idx, file in enumerate(args.file):
                idx = idx if args.label_f else None
                feature_file = Path(file)
                if args.raa is None:
                    raa = list(feature_file.name.split('-')[-1])
                else:
                    raa = list(args.raa)
                ul.extract_to_file(feature_file, out, raa,k, gap, lam, 
                                   label=idx, mode="a+")
        else:
            for idx, (file, out) in enumerate(zip(args.file, args.output)):
                feature_file = Path(file)
                if args.raa is None:
                    raa = list(feature_file.name.split('-')[-1])
                else:
                    raa = list(args.raa)
                label = idx if args.label_f else None
                ul.extract_to_file(feature_file, Path(out), raa, k, gap, lam, label=label, mode="w")

def sub_hpo(args):
    params = ul.param_grid(args.C, args.gamma)
    x, y = ul.load_normal_data(args.file)
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
        x, y = ul.load_normal_data(args.file)
        model = cp.train(x, y, c, g)
        cp.save_model(model, args.output) 

def sub_predict(args):
    x, _ = ul.load_normal_data(args.file, label_exist=False)
    model = ul.load_model(args.model)
    y_pred = cp.predict(x, model)
    ul.save_y(args.output, y_pred)
    
def sub_eval(args):
    c, g, cv = args.C, args.gamma, args.cv
    if args.directory:
        out = Path(args.output)
        out.mkdir(exist_ok=True)
        metric_dic = cp.batch_evaluate(Path(args.file), out, c, g, cv, args.process)
        result_json = args.output + ".json"
        ul.save_json(metric_dic, result_json)
    else:
        x, y = ul.load_normal_data(args.file)
        metric, cm, labels = cp.evaluate(x, y, c, g, cv)
        ul.save_report(metric, cm, labels, args.output)        

def sub_roc(args):
    model = ul.load_model(args.model)
    x, y = ul.load_normal_data(args.file)
    ul.roc_eval(x, y, model, args.output)

def sub_ifs(args):
    ul.exist_file(*args.file)
    C, gamma, step, cv, n_jobs = args.C, args.gamma, args.step, args.cv, args.process
    if args.mix:
        x, y = ul.feature_mix(args.file)
        noraml_x, noraml_y = ul.load_normal_data((x, y))
        result_ls = cp.feature_select(noraml_x, noraml_y, C, gamma, step, cv, n_jobs)
        x_tricks = [0] + [i for i in range(0, x.shape[1], args.step)][1:]
        x_tricks.append(x.shape[1])
        acc_ls = [0] + [i[0][0][0] for i in result_ls]
        ul.save_y(args.output, x_tricks, acc_ls)
        max_acc = max(acc_ls)
        best_n = acc_ls.index(max_acc) * step
        draw.p_fs(x_tricks, acc_ls, args.output[0]+'.png', max_acc=max_acc, best_n=best_n)
    else:
        for file, out in zip(args.file, args.output): 
            noraml_x, noraml_y = ul.load_normal_data(file)
            result_ls = cp.feature_select(noraml_x, noraml_y, C, gamma, step, cv, n_jobs)
            x_tricks = [0] + [i for i in range(0, noraml_x.shape[1], args.step)][1:]
            x_tricks.append(noraml_x.shape[1])
            acc_ls = [0] + [i[0][0][0] for i in result_ls]
            ul.save_y(out, x_tricks, acc_ls)
            max_acc = max(acc_ls)
            best_n = acc_ls.index(max_acc) * step
            draw.p_fs(x_tricks, acc_ls, out+'.png', max_acc=max_acc, best_n=best_n)

def sub_plot(args):
    ul.mkdirs(args.outdir)
    with open(args.file, 'r') as f:
        re_dic = json.load(f)
        ul.eval_plot(re_dic, Path(args.outdir), fmt=args.format)

def sub_merge(args):
    mix_data = ul.merge_feature_file(args.label, args.file)
    ul.write_array(mix_data, args.output)

def sub_split(args):
    ts = args.testsize
    if 0 < ts < 1:
        data_train, data_test = ul.split_data(args.file, ts)
        ul.write_array(data_train, f"{1-ts}_{args.output}")
        ul.write_array(data_test, f"{ts}_{args.output}")
    else:
        print("error")

# TODO
def sub_transfer(args):
    pass
        
def command_parser():
    parser = argparse.ArgumentParser(description='reduce amino acids toolkit')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_v = subparsers.add_parser('view', help='view the reduce amino acids scheme')
    parser_v.add_argument('-t', '--type', nargs='+', type=int, required=True, 
                          choices=list([i for i in range(1, 74)]),help='type id')
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
    parser_ex.add_argument('-raa', help='reduced amino acid cluster')
    parser_ex.add_argument('-m', '--merge', action='store_true', help='merge feature files into one')
    parser_ex.add_argument('-o', '--output', nargs='+', required=True, help='output directory')
    parser_ex.add_argument('-p', '--process',type=int, choices=list([i for i in range(1, os.cpu_count())]),
                                 default=1, help='cpu number')
    parser_ex.add_argument('--label-f', action='store_false', help='feature label')
    parser_ex.set_defaults(func=sub_extract)

    parser_ex = subparsers.add_parser('hpo', help='hpyper-parameter optimization')
    parser_ex.add_argument('file', help='feature file for hpyper-parameter optimization')
    parser_ex.add_argument('-c', '--C',  nargs='+', required=True, type=float,
                            help='regularization parameter value range [start, stop, [num]]')
    parser_ex.add_argument('-g', '--gamma', nargs='+', required=True, type=float,
                            help='Kernel coefficient value range [start, stop, [num]]')
    parser_ex.set_defaults(func=sub_hpo)
    
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

