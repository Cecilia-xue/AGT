import os
import torch
from data.hsi import HSI
from data.cifar import cifar
from data.transforms_builder import transforms_builder


def data_builder(args):

    if args.datatype == 'HSI':
        data_path = os.path.join(args.root, args.datatype)
        transforms = transforms_builder(args)
        train_set = HSI(data_path, args.dataset, args.patch_size, args.sample_list, is_train=True, transforms=transforms)
        val_set = HSI(data_path, args.dataset, args.patch_size, args.sample_list, is_train=False, transforms=None)
        num_training_steps_per_epoch = len(train_set) // args.batch_size
    elif args.datatype == 'PHSI':
        data_path = os.path.join(args.root, 'CIFAR')
        train_set = cifar(data_path, args.dataset, is_train=True)
        val_set = cifar(data_path, args.dataset, is_train=False)
        num_training_steps_per_epoch = len(train_set) // args.batch_size

    return train_set, val_set, num_training_steps_per_epoch


