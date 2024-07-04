import matplotlib.pyplot as plt
import numpy as np

def spectrum_wise_normalization(data):
    max_num = data.max()
    min_num = data.min()
    nl_data = (data-min_num) / (max_num-min_num)

    return nl_data


data_dir = '/home/disk/data/HSI/HoustonU/HoustonU_data.npy'
# data_dir = '/home/disk/data/HSI/PaviaU/PaviaU_data.npy'


data = np.load(data_dir)
min_arr = data.min(axis=(0, 1))
max_arr = data.max(axis=(0, 1))
print('done')







