import os
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt



def convert(dataset, dataset_HSI, dataset_gt):
    dataset_mat_dir = data_mat_dir + '{}/{}.mat'.format(dataset, dataset)
    dataset_gt_dir = data_mat_dir + '{}/{}_gt.mat'.format(dataset, dataset)

    if dataset in ['Indian_pines', 'PaviaU', 'KSC','Salinas']:
        HSI_data = sio.loadmat(dataset_mat_dir)[dataset_HSI]
        HSI_gt = sio.loadmat(dataset_gt_dir)[dataset_gt]

    elif dataset == 'HoustonU':
        HSI_data = h5py.File(dataset_mat_dir)[dataset_HSI][:]
        HSI_data = HSI_data.transpose(1,2,0)
        HSI_gt = h5py.File(dataset_gt_dir)[dataset_gt][:]
    elif dataset == 'PaivaC':
        print('done')
        # HSI_data = h5py.File(dataset_mat_dir)[dataset_HSI][:]
        # HSI_data = HSI_data.transpose(1, 2, 0)
        # HSI_gt = h5py.File(dataset_gt_dir)[dataset_gt][:]


    np.save(dataset_mat_dir.replace('.mat', '_data.npy'), HSI_data)
    np.save(dataset_mat_dir.replace('.mat', '_label.npy'), HSI_gt)

    print('{} convert done!'.format(dataset))

if __name__ == "__main__":

    data_mat_dir = '/home/disk1/data/HSI/'
    dataset_mat_dir = data_mat_dir + 'PaviaC/PaviaC.mat'
    dataset_gt_dir = data_mat_dir + 'PaviaC/PaviaC_gt.mat'

    HSI_data = sio.loadmat(dataset_mat_dir)['pavia']
    HSI_gt = sio.loadmat(dataset_gt_dir)['pavia_gt']

    np.save(dataset_mat_dir.replace('.mat', '_data.npy'), HSI_data)
    np.save(dataset_mat_dir.replace('.mat', '_label.npy'), HSI_gt)

    print('PaivaC convert done')









