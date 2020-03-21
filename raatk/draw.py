# -*- coding: utf-8 -*-

"""
draw.py
~~~~~~~~~~~~
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def p_univariate_density(acc, out):
    fig, ax = plt.subplots()
    sns.set_style('whitegrid')
    ax = sns.kdeplot(acc, ax=ax)
    ax.set_xlim(40, 100)
    ax.set_xticks(range(40, 100, 10))
    ax.set_xlabel('ACC')
    plt.savefig(out, dpi=1000, bbox_inches="tight")

def p_bivariate_density(x, y, out):
    g = sns.jointplot(x, y, kind="kde", xlim=(2, 20), ylim=(0, 1))
    g.set_axis_labels("size", "Acc")
    g.ax_joint.set_xticks(range(2, 21))
    plt.savefig(out, dpi=1000, bbox_inches="tight")

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
    plt.savefig(out, dpi=1000, bbox_inches="tight")


# TODO
def p_acc_heat(data, vmin, vmax, xticklabels, out, **kwargs):
    annot_size = kwargs.get("annot_size", 2)
    tick_size = kwargs.get("tick_size", 3)
    label_size = kwargs.get("label_size", 4)
    mask = data <= 0
    ax = sns.heatmap(data*100, vmin=vmin*100, vmax=vmax*100, cmap='RdYlGn_r', 
                      mask=mask, annot=True, fmt=".1f", robust=True,
                      annot_kws={"color": "black", "fontsize": annot_size}, 
                      xticklabels=True)
    ax.set_ylim([0, 19])
    ax.set_xlim([0, len(xticklabels)])
    ax.set_xticklabels(xticklabels, fontsize=tick_size, rotation=45, ha="center",va="center")
    ax.set_yticklabels(range(2,21), fontsize=tick_size)
    ax.set_xlabel('Type', fontsize=label_size)
    ax.set_ylabel('Cluster Size', fontsize=label_size)
    plt.savefig(out, dpi=1000, bbox_inches="tight")
    plt.close()

def p_fs(x_trick, y_trick, out, **kwargs):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_trick = x_trick[:len(y_trick)]
    plt.plot(x_trick, y_trick)
    max_acc = kwargs.get("max_acc", None)
    best_n = kwargs.get("best_n", None)
    if max_acc and best_n:
        plt.scatter(best_n, max_acc, marker='*', c='r')
        plt.text(best_n, max_acc + 0.002, f'{best_n}, {max_acc:.4f}',
                ha='center', va='bottom', fontsize=6, fontweight='bold')
    plt.savefig(out, dpi=1000)
