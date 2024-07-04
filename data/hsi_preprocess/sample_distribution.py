import os
import numpy as np

def distribution(data_npy_dir, dataset, sample_num, tag):

    train_info_arr = []
    test_info_arr = []

    label_map = np.load(os.path.join(data_npy_dir, '{}/{}_label.npy'.format(dataset, dataset)))
    category_num = label_map.max()
    for i in range(1, category_num+1):
        index_i = np.where(label_map==i)
        labeled_sample_num = len(index_i[0])
        label_arr = np.zeros(labeled_sample_num) + i
        shuffle_index = np.random.permutation(labeled_sample_num)
        train_num = min(sample_num, labeled_sample_num // 2) if not isinstance(sample_num, np.ndarray) else min(sample_num[i-1], labeled_sample_num // 2)
        train_coord, test_coord = shuffle_index[:train_num], shuffle_index[train_num:]
        train_coord_x, train_coord_y, train_labels = index_i[0][train_coord], index_i[1][train_coord], label_arr[:train_num]
        test_coord_x, test_coord_y, test_labels = index_i[0][test_coord], index_i[1][test_coord], label_arr[train_num:]

        train_info = np.vstack((train_coord_x, train_coord_y, train_labels)).astype(np.int)
        test_info = np.vstack((test_coord_x, test_coord_y, test_labels)).astype(np.int)

        train_info_arr.append(train_info.transpose())
        test_info_arr.append(test_info.transpose())

    train_info_arr, test_info_arr = np.vstack(train_info_arr), np.vstack(test_info_arr)

    sample_list_dir = os.path.join(data_npy_dir, '{}/sample_list'.format(dataset))
    if not os.path.exists(sample_list_dir): os.makedirs(sample_list_dir)

    if tag == 'ab':
        train_list_dir = os.path.join(sample_list_dir, 'train_ab_{}.txt'.format(train_sample_num))
        test_list_dir = os.path.join(sample_list_dir, 'test_ab_{}.txt'.format(train_sample_num))
    elif tag == 'cp':
        train_list_dir = os.path.join(sample_list_dir, 'train_cp.txt')
        test_list_dir = os.path.join(sample_list_dir, 'test_cp.txt')

    with open(train_list_dir, 'w') as f:
        f.write('coord_x, coord_y, label\n')
        for info_item in train_info_arr:
            f.write(' '.join(list(map(str, info_item))) + '\n')

    with open(test_list_dir, 'w') as f:
        f.write('coord_x, coord_y, label')
        for info_item in test_info_arr:
            f.write(' '.join(list(map(str, info_item))) + '\n')

    print('sample distribution for {} done !'.format(dataset))


if __name__ == "__main__":

    # datasets = ['KSC', 'PaviaU', 'Indian_pines', 'HoustonU'，‘Salinas’]

    data_npy_dir = '/home/disk1/data/HSI/'
    datasets = ['HoustonU', 'PaviaU', 'Indian_pines']

    # ablation study settings in transfer learning experiments
    tag = 'ab'  # [ab, cp] [ablation study, comparison]
    train_sample_num = 75
    for dataset in datasets:
        distribution(data_npy_dir, dataset, train_sample_num, tag)

    train_sample_num = 50
    for dataset in datasets:
        distribution(data_npy_dir, dataset, train_sample_num, tag)

    train_sample_num = 25
    for dataset in datasets:
        distribution(data_npy_dir, dataset, train_sample_num, tag)


    # sample distribution settings in comparison experiments
    datasets = ['HoustonU', 'PaviaU','PaviaC']

    train_sample_num = 150
    tag='cp'
    for dataset in datasets:
        distribution(data_npy_dir, dataset, train_sample_num, tag)

    datasets = ['Indian_pines']
                                # 1  2    3    4   5   6   7  8  9   10    11  12    13  14  15 16
    train_sample_num = np.array([10,150, 150, 150, 10,150,10,150,10, 150, 150, 150, 150,150,10,10])
    for dataset in datasets:
        distribution(data_npy_dir, dataset, train_sample_num, tag)

    # pretrained dataset
    datasets = ['Salinas']
    train_sample_num = 2000
    tag = 'cp'
    for dataset in datasets:
        distribution(data_npy_dir, dataset, train_sample_num, tag)



