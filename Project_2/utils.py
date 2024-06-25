import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import logsumexp
import itertools


def load_data(file_name, data_path='data/'):
    return pd.read_csv(data_path + file_name + '.csv', index_col=0).values


def accuracy(true, pred):
    return np.mean(true == pred)


def plot_barplot(a, b, labels, xlabel=None, ylabel=None, title=None, figname=None, ylim=(0,1.1)):
    fig, ax = plt.subplots(1, 1)
    temp = pd.DataFrame([a, b], columns=np.arange(len(a)), index=labels).T
    temp.plot.bar(rot=0, width=.5, ax=ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(f'plots/{figname}.png')
    plt.show()


def plot_with_errorbars(data, x_ticks, ax, label=None, title=None, xlabel=None, ylabel=None,
                        ylims=None, c='black'):
    mean_data = data.mean(axis=0)
    std_data = data.std(axis=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(x_ticks, mean_data, label=label, c=c, marker='o')
    ax.errorbar(x_ticks, mean_data, std_data, c=c,
                capsize=3, capthick=1)
    if ylims is not None: ax.set_ylim(ylims)
    return ax


def plot_posterior_vs_M(est_posteriors, exact_posteriors, algo_name, M_arr, t=5, x=1):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        axs.flat[i].plot(M_arr, [exact_posteriors[i, t, x]] * len(M_arr), label='Exact', c='tab:orange', marker='o')
        plot_with_errorbars(est_posteriors[:, :, i, t, x], x_ticks=M_arr, ax=axs.flat[i],
                            label=algo_name, title=f'obs[{i}]',
                            xlabel='M', ylabel=f'posterior - p(X{t}={x} | obs[{i}])',
                            ylims=(-0.1, 1.1))
    axs.flat[0].legend()
    fig.suptitle(f'{algo_name} vs Exact posterior for t={t}')
    fig.tight_layout()
    plt.savefig(f'plots/{algo_name}_est_vs_exact_posterior.png')
    plt.show()
