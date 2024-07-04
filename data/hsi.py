import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import numpy as np
import os


def max_min_normalization(data):
    max_num = data.max()
    min_num = data.min()
    nl_data = (data-min_num) / (max_num-min_num)

    return nl_data

def max_normalization(data):
    nl_data = data/data.max()

    return nl_data

def standardize(data):
    mean, std=data.mean(), data.std()
    nl_data = (data-mean) / std

    return nl_data


class HSI(data.Dataset):
    def __init__(self, data_path, dataset, patch_size, sample_list, is_train=False, transforms=None):
        data = np.load(os.path.join(data_path, '{}/{}_data.npy'.format(dataset, dataset)))
        # data = data/data.max()
        data = standardize(data)
        H, W, S = data.shape
        data_padded = np.zeros((H+patch_size, W+patch_size, S))
        self.pad_size = patch_size // 2
        data_padded[self.pad_size:self.pad_size+H, self.pad_size:self.pad_size+W, :] = data
        self.data=data_padded
        self.patch_size = patch_size
        self.transforms = transforms

        if is_train:
            with open(os.path.join(data_path, '{}/sample_list/train_{}.txt'.format(dataset, sample_list)), 'r') as f:
                sample_list = f.readlines()
        else:
            with open(os.path.join(data_path, '{}/sample_list/test_{}.txt'.format(dataset, sample_list)), 'r') as f:
                sample_list = f.readlines()

        self.sample_list=[item.rstrip('\n') for item in sample_list[1:]]
        self.length = self.sample_list.__len__()

    def __getitem__(self, index):
        loc_x, loc_y, label = np.array(self.sample_list[index].split(), np.int)
        sample = self.data[loc_x:loc_x+self.patch_size, loc_y:loc_y+self.patch_size, :]
        sample = torch.from_numpy(sample).float() # [h, w, l]
        sample = sample.permute(2, 0, 1) # [l, h, w]

        label = label-1
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample = sample.unsqueeze(dim=0)

        return sample, label

    def __len__(self):
        return self.length

