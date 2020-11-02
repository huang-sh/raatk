# -*- coding: utf-8 -*-

"""
draw.py
~~~~~~~~~~~~
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go


def p_univariate_density(acc, label="ACC", key='Acc', out=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax = ax
    sns.set_style('whitegrid')
    ax = sns.kdeplot(acc, ax=ax, label=label)
    ax.set_xlim(40, 100)
    ax.set_xticks(range(40, 100, 10))
    ax.set_xlabel(key)
    if out:
        plt.savefig(out, dpi=1000, bbox_inches="tight")
    else:
        pass


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
    mask = data <= 0
    sns.set(font_scale=0.3)
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax, cmap='RdYlGn_r', 
                      mask=mask, annot=True, fmt=".1f", robust=True,
                      annot_kws={"color": "black"}, xticklabels=True)
    ax.set_ylim([0, 19])
    ax.set_xlim([0, len(xticklabels)])
    ax.set_xticklabels(xticklabels, rotation=45, ha="center",va="center")
    ax.set_yticklabels(range(2,21))
    ax.set_xlabel('Type')
    ax.set_ylabel('Cluster Size')
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


def visual_feature(x, y, out, labels=None):
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'blue', 'grey', 'red']
    labels = labels if labels else np.unique(y).astype(int)
    for color, i, target_name in zip(colors, np.unique(y), labels):
        plt.scatter(x[y == i, 0], x[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.savefig(out, dpi=1000)


def visual_feature_3d(x, y, out, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    markers = ['o', '^', '1', '*', '+', 'x', '.','D']
    for i in np.unique(y).astype(int):
        x1, x2, x3 = x[y == i, 0], x[y == i, 1], x[y == i, 2]
        ax.scatter(x1, x2, x3, marker=markers[i], label=i)
    ax.legend(loc='best')
    plt.savefig(out, dpi=1000)


def sankey(label, source, target, value, out, title):
    fig = go.Figure(
        data=[
            go.Sankey(
                customdata=label,
                orientation='h',
                node=dict(
                  pad=15,
                  thickness=20,
                  line=dict(color="black", width=0.5),
                  label=label,
                  hovertemplate='%{label}'
                        ),
                link=dict(
                  source=source,
                  target=target,
                  value=value
                            ),
                    ),
        ])
    fig.update_layout(title_text=f"Reduction alphabet of {title}", font_size=10)
    fig.write_html(f'{out}.html')
