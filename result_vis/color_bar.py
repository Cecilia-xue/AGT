import numpy as np
from result_vis.color_dict import color_dict
from PIL import Image

datasets = ['Indian_pines', 'PaviaU', 'HoustonU']
categroy_nums = [16, 9, 15]
splits = [4, 2, 4]
gt_map_paths = [
    '/home/disk/data/HSI/Indian_pines/Indian_pines_label.npy',
    '/home/disk/data/HSI/PaviaU/PaviaU_label.npy',
    '/home/disk/data/HSI/HoustonU/HoustonU_label.npy'
]

for dataset, category_num, split, in zip(datasets, categroy_nums, splits):
    color_map = color_dict[dataset]
    color_bar = []
    for i in range(1, category_num+1):
        bar_i = np.zeros([40,40,3], np.uint8)
        bar_i[:,:] = np.array(color_map[str(i)], np.uint8)
        color_bar.append(bar_i)

        if i!=0 and i% split == 0 or i==category_num:
            color_bar = np.vstack(color_bar)
            color_bar_im = Image.fromarray(color_bar)
            color_bar_im.save('./color_bar_vis/{}_{}.jpg'.format(dataset, i))
            color_bar = []





for dataset, path in zip(datasets, gt_map_paths):
    color_map = color_dict[dataset]
    label_map = np.load(path)
    h, w = label_map.shape
    label_color_map = np.zeros([h, w, 3], np.uint8)
    for i in range(h):
        for j in range(w):
            label_color_map[i,j] = np.array(color_map[str(label_map[i,j])], np.uint8)

    label_color_im = Image.fromarray(label_color_map)
    label_color_im.save('./color_bar_vis/{}_label_map.png'.format(dataset))





