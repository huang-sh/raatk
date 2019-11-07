import os
import glob
import json
import time
import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colorbar
from sklearn.metrics import roc_curve, roc_auc_score


def p_roc(y_test, y_pro, out):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for v in y_pro:
        fp, tp, _ = roc_curve(y_test, y_pro[v])
        auc = roc_auc_score(y_test, y_pro[v])
        plt.plot(fp, tp, label=f'{v} (AUC = {auc:.2f})')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(f'{out}_plot/roc_comparison.svg', dpi=1000, bbox_inches="tight")


def p_univariate_density(acc_ls, lable, out):
    fig, ax = plt.subplots()
    sns.set_style('whitegrid')
    for acc, l in zip(acc_ls, lable):
        acc = [i * 100 for i in acc if i > 0]
        ax = sns.kdeplot(np.array(acc), label=l, ax=ax)
        ax.set_xlim(40, 100)
        ax.set_xticks(range(40, 100, 10))
        ax.set_xlabel('ACC')
    plt.savefig(out, dpi=1000, bbox_inches="tight")


def p_bivariate_density(x, y, label, out):
    g = sns.jointplot(x, y, kind="kde", xlim=(2, 20), ylim=(0, 1))
    g.set_axis_labels("size", "Acc")
    g.ax_joint.set_xticks(range(2, 21))
    plt.savefig(out, dpi=600, bbox_inches="tight")

def p_comparison_type(diff_size, same_size, types_label, out):
    grid = plt.GridSpec(1, 4,)
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(grid[0, :2])  # same type
    ax2 = plt.subplot(grid[0, 2:])  # same size

    sizes_acc = diff_size[diff_size > 0]
    sizes_label = np.where(diff_size > 0)[0] + 2
    max_acc = sizes_acc.max()
    m_size_idx = np.where(sizes_acc == max_acc)[0][0] + 1
    ax1.bar(range(1, len(sizes_acc)+1), sizes_acc, color='grey')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Size',)
    ax1.set_xticks(range(1, len(sizes_acc)+1))
    ax1.set_xticklabels(sizes_label, fontdict={'Size': 12})
    ax1.set_ylabel('Acc', fontdict={'Size': 15})
    ax1.text(m_size_idx, max_acc+0.01, f'{max_acc:.3f}', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

    types_acc = same_size[same_size > 0]
    types_label = np.array(types_label)[same_size > 0]
    m_size_idx = np.where(types_acc == max_acc)[0][0] + 1
    ax2.bar(range(1, len(types_acc)+1), types_acc, color='grey')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Type')
    ax2.set_xticks(range(1, len(types_acc)+1))
    ax2.set_xticklabels(types_label, fontsize=8, rotation=-35)  #
    ax2.text(m_size_idx, max_acc+0.01, f'{max_acc:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")


# def p_acc_heat(data, vmin, vmax, xticklabels, out):
#     ck = data.shape
#     fig = plt.figure(figsize=(22, 44))
#     ax = plt.subplot(1, 1, 1)
#     norm = BoundaryNorm(boundaries=np.linspace(vmin, vmax, 41), ncolors=256)
#     # cmap = LinearSegmentedColormap.from_list('mycmap', ['lightgreen', 'yellow', 'red'])
#     im = ax.imshow(data, cmap='RdYlGn_r', norm=norm)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="1.5%", pad=0.05)
#     # cbar.set_clim(0.5, 1.0)
#     plt.colorbar(im, cax=cax)
#     ax.set_xticks(np.arange(ck[1]))
#     ax.set_xticklabels(xticklabels, fontsize=10)
#     ax.set_yticks(np.arange(ck[0]))
#     ax.set_yticklabels(range(2, 21), fontsize=10)
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#     for i in range(ck[0]):
#         for j in range(ck[1]):
#             if data[i, j] >= -3:
#                 acc = '%.2f' % (data[i, j] * 100)
#                 if ck[1] > 45:
#                     ax.text(j, i, acc, ha="center", va="center", color="k", fontsize=6,)
#                 else:
#                     ax.text(j, i, acc, ha="center", va="center", color="k", fontsize=11, )
#     fig.tight_layout()
#     plt.savefig(out, dpi=600, bbox_inches="tight")

# TODO
def p_acc_heat(data, vmin, vmax, xticklabels, out):
    cr = data.shape
    mask = data <= 0
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax, cmap='RdYlGn_r', mask=mask, annot=True, fmt=".4",
                      annot_kws={"fontsize":5, "color": "black"}, xticklabels=xticklabels, yticklabels=range(2, 21))
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel('Type', fontsize=8)
    ax.set_ylabel('Cluster Size', fontsize=8)
    plt.savefig(out, dpi=600, bbox_inches="tight")

def p_roc_al(param, out):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for clf, metrics in param.items():
        plt.plot(metrics[0], metrics[1], label=f'{clf} (AUC = {metrics[-1]:.2f})')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.savefig(out, dpi=600)

def p_fs(score_ls, out):
    plt.figure()
    plt.plot(range(1, len(score_ls) + 1), score_ls)
    max_acc = max(score_ls)
    best_n = score_ls.index(max_acc)
    plt.scatter(best_n, max_acc, marker='*', c='r')
    plt.text(best_n, max_acc + 0.002, f'{best_n}, {max_acc:.4f}',
             ha='center', va='bottom', fontsize=6, fontweight='bold')
    plt.savefig(out, dpi=600)
