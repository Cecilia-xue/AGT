import os
import pickle
import numpy as np
from PIL import Image
import torchvision.datasets.cifar

base_folder = 'cifar-100-python'
url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
train_list = [
    ['train', '16019d7e3df5f24257cddd939b257f8d'],
]

test_list = [
    ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
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

    for i in range(100):
        path = os.path.join(save_dir, set, str(i))
        if not os.path.exists(path): os.makedirs(path)
    count = np.zeros(100, np.int)
    with open(os.path.join(save_dir, '{}_list.txt'.format(set)), 'w') as f:
        for img, label in zip(data, labels):
            rgb = img.reshape(3, 32, 32).transpose(1, 2, 0)
            rgb_name = '{}'.format(count[label]).zfill(5) + '.jpg'
            img_save_dir = os.path.join(save_dir, set, str(label), rgb_name)
            count[label] += 1
            im = Image.fromarray(rgb)
            im.save(img_save_dir)
            info = os.path.join('{}'.format(set), str(label), rgb_name) + ' {}\n'.format(label)
            f.write(info)
            print(img_save_dir)

if __name__ == '__main__':
    cifar10_dir = '/home/disk1/data/CIFAR'
    save_dir = '/home/disk1/data/CIFAR/cifar100_rgb'

    convert(cifar10_dir, save_dir, train_list, 'train')
    convert(cifar10_dir, save_dir, test_list, 'test')


