import numpy as np
from PIL import Image
from result_vis.color_dict import color_dict

datasets = ['Indian_pines', 'PaviaU', 'HoustonU']
pred_map_paths = [
    './pre_maps/Indian_pines_map.txt',
    './pre_maps/PaviaU_map.txt',
    './pre_maps/HoustonU_map.txt'
]
gt_map_paths = [
    '/home/disk/data/HSI/Indian_pines/Indian_pines_label.npy',
    '/home/disk/data/HSI/PaviaU/PaviaU_label.npy',
    '/home/disk/data/HSI/HoustonU/HoustonU_label.npy'
]

for dataset, pred_map_path, gt_map_path in zip(datasets, pred_map_paths, gt_map_paths):
    color_map = color_dict[dataset]
    gt_map = np.load(gt_map_path)

    with open(pred_map_path, 'r') as f:
        pred_info = f.readlines()
        H, W = int(pred_info[0].strip().split(':')[1]), int(pred_info[1].strip().split(':')[1])
        pred_map = np.array([item.strip() for item in pred_info[2:]]).reshape(H, W)

        pred_map_color = np.zeros([H, W, 3], np.uint8)
        for i in range(H):
            for j in range(W):
                if gt_map[i,j]>0:
                    pred_map_color[i,j] = np.array(color_map[str(int(pred_map[i,j])+1)], np.uint8)

        pred_map_im = Image.fromarray(pred_map_color)
        pred_map_im.save('./pre_color_vis/{}_pred_map.png'.format(dataset))





