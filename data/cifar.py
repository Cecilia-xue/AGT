import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import numpy as np
import os

class cifar(data.Dataset):
    def __init__(self, data_path, dataset, is_train=False):
        self.data_root = os.path.join(data_path, '{}_phsi'.format(dataset))
        if is_train:
            sample_list_dir = os.path.join(data_path, '{}_phsi'.format(dataset), 'train_list.txt')
            with open(sample_list_dir, 'r') as f:
                sample_list = f.readlines()
        else:
            sample_list_dir = os.path.join(data_path, '{}_phsi'.format(dataset), 'test_list.txt')
            with open(sample_list_dir, 'r') as f:
                sample_list = f.readlines()

        self.sample_list = [line.rstrip() for line in sample_list]
        self.length = self.sample_list.__len__()

    def __getitem__(self, index):
        sample_info = self.sample_list[index]
        sample_dir, label = sample_info.split()

        sample =  torch.from_numpy(np.load(os.path.join(self.data_root, sample_dir))).unsqueeze(dim=0)
        label = int(label)
        return sample, label

    def __len__(self):
        return self.length

