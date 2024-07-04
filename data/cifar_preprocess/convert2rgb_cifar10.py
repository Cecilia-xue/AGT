import os
import pickle
import numpy as np
from PIL import Image
import torchvision.datasets.cifar

base_folder = 'cifar-10-batches-py'
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
train_list = [
    ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
]

test_list = [
    ['test_batch', '40351d587109b95175f43aff81a1287e'],
]


def convert(cifar10_dir, save_dir, data_list, set='train'):
    # training set
    data, labels = [], []
    for file_name, checksum in data_list:
        file_path = os.path.join(cifar10_dir, base_folder, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'labels' in entry:
                labels.extend(entry['labels'])
            else:
                labels.extend(entry['fine_labels'])
    data = np.vstack(data)

    for i in range(10):
        path = os.path.join(save_dir, set, str(i))
        if not os.path.exists(path): os.makedirs(path)
    count = np.zeros(10, np.int)
    with open(os.path.join(save_dir, '{}_list.txt'.format(set)), 'w') as f:
        for img, label in zip(data, labels):
            rgb = img.reshape(3, 32, 32).transpose(1, 2, 0)
            rgb_name = '{}'.format(count[label]).zfill(5) + '.jpg'
            img_save_dir = os.path.join(save_dir, set, str(label), rgb_name)
            count[label] += 1
            im = Image.fromarray(rgb)
            im.save(img_save_dir)
            info = os.path.join(set, str(label), rgb_name) + ' {}\n'.format(label)
            f.write(info)
            print(img_save_dir)

if __name__ == '__main__':
    cifar10_dir = '/home/disk1/data/CIFAR'
    save_dir = '/home/disk1/data/CIFAR/cifar10_rgb'

    convert(cifar10_dir, save_dir, train_list, 'train')
    convert(cifar10_dir, save_dir, test_list, 'test')


