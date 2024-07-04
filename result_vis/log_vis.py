import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

import scipy

def log_info_get(log_file):
    with open(log_file, 'r') as f:
        log_info = f.readlines()

    train_acc, train_epoch = [], []
    val_acc, val_epoch = [], []

    for info in log_info:
        if "samples" in info:
            info_list = info.split()
            epoch = int(info_list[3])
            acc = float(info_list[8].split(':')[1])
            val_epoch.append(epoch)
            val_acc.append(acc)
        else:
            info_list = info.split()
            epoch = int(info_list[2].split(':')[1])
            acc = float(info_list[8].split(':')[1])
            train_epoch.append(epoch)
            train_acc.append(acc)

    train_acc_arr, train_epoch_arr = np.array(train_acc), np.array(train_epoch)
    val_acc_arr, val_epoch_arr = np.array(val_acc), np.array(val_epoch)

    return train_acc_arr, train_epoch_arr, val_acc_arr, val_epoch_arr

def logs_info_get(log_files):
    train_acc_arrs = []
    train_epoch_arrs = []
    val_acc_arrs = []
    val_epoch_arrs = []
    for log_file in log_files:
        train_acc_arr, train_epoch_arr, val_acc_arr, val_epoch_arr = log_info_get(log_file)
        train_acc_arrs.append(train_acc_arr)
        train_epoch_arrs.append(train_epoch_arr)
        val_acc_arrs.append(val_acc_arr)
        val_epoch_arrs.append(val_epoch_arr)

    return train_acc_arrs, train_epoch_arrs, val_acc_arrs, val_epoch_arrs


def train_res_plot(epoch_arrs, acc_arrs, save_dir, region, log_labels, end_epoch):
    plt.rcParams['font.size'] = 16
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for epoch, acc in zip(epoch_arrs, acc_arrs):
        acc = scipy.signal.savgol_filter(acc, 24, 1)
        ax.plot(epoch[:end_epoch], acc[:end_epoch])
    ax.legend(labels=log_labels)

    axins = ax.inset_axes((0.2, 0.2, 0.5, 0.5))
    for epoch, acc in zip(epoch_arrs, acc_arrs):
        acc = scipy.signal.savgol_filter(acc, 24, 1)
        axins.plot(epoch[region[0]:region[1]], acc[region[0]:region[1]])
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

    plt.savefig(save_dir + '/train.png'.format(type))
    plt.close()


def val_res_plot(epoch_arrs, acc_arrs, save_dir, region, log_labels, end_epoch):
    plt.rcParams['font.size'] = 16
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for epoch, acc in zip(epoch_arrs, acc_arrs):
        acc = scipy.signal.savgol_filter(acc, 24, 1)
        ax.plot(epoch[:30], acc[:30])
    ax.legend(labels=log_labels)

    # axins = ax.inset_axes((0.2, 0.2, 0.5, 0.5))
    # for epoch, acc in zip(epoch_arrs, acc_arrs):
    #     acc = scipy.signal.savgol_filter(acc, 24, 1)
    #     axins.plot(epoch[region[0]:region[1]], acc[region[0]:region[1]])
    #     mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

    plt.savefig(save_dir + '/val.png'.format(type))
    plt.close()


if __name__ == "__main__":
    log_files = [
        './logs/Indian_50.txt',
        './logs/Indian_50_ft.txt',
        './logs/Indian_50_lora.txt',
        './logs/Indian_50_clip.txt',
        './logs/Indian_50_pit.txt'
    ]

    log_labels = ['Base', 'Fine-tune', 'LORA', 'Clip-adapter', 'SDT']
    save_dir = './log_vis'

    train_acc_arrs, train_epoch_arrs, val_acc_arrs, val_epoch_arrs = logs_info_get(log_files)

    region = [15, 65]
    end_epoch = 150

    train_res_plot(train_epoch_arrs, train_acc_arrs, save_dir, region, log_labels, end_epoch)

    region = [3, 15]
    end_epoch = 100

    val_res_plot(val_epoch_arrs, val_acc_arrs, save_dir, region, log_labels, end_epoch)



















