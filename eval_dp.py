import torch
import numpy as np
import torch.utils.data as data
from sklearn.metrics import cohen_kappa_score
import os
import argparse
from datasets_info import datasets_info
from torch.nn import DataParallel
from models.model_builder import model_builder
import warnings
import time
warnings.filterwarnings("ignore")
from tqdm import tqdm

def OA_AA_K_cal(pre_label, tar_label):
    acc=[]
    samples_num = len(tar_label)
    category_num=tar_label.max()+1
    for i in range(category_num):
        loc_i = np.where(tar_label==i)
        OA_i = np.array(pre_label[loc_i]==tar_label[loc_i], np.float32).sum()/len(loc_i[0])
        acc.append(OA_i)

    OA = np.array(pre_label==tar_label, np.float32).sum()/samples_num
    AA = np.average(np.array(acc))
    K = cohen_kappa_score(tar_label, pre_label)
    acc.append(OA)
    acc.append(AA)
    acc.append(K)
    return np.array(acc)


def standardize(data):
    mean, std=data.mean(), data.std()
    nl_data = (data-mean) / std

    return nl_data

class HSI(data.Dataset):
    def __init__(self, dataset_path, patch_size):
        data = np.load(dataset_path)
        data = standardize(data)
        H, W, S = data.shape
        self.H, self.W = H, W
        data_padded = np.zeros((H+patch_size, W+patch_size, S))
        self.pad_size = patch_size // 2
        data_padded[self.pad_size:self.pad_size+H, self.pad_size:self.pad_size+W, :] = data
        self.data=data_padded
        self.patch_size = patch_size
        self.sample_list = []
        for i in range(H):
            for j in range(W):
                self.sample_list.append([i,j])
        self.length = self.sample_list.__len__()


    def __getitem__(self, index):
        loc_x, loc_y = self.sample_list[index]
        sample = self.data[loc_x:loc_x+self.patch_size, loc_y:loc_y+self.patch_size, :]
        sample = torch.from_numpy(sample).float() # [h, w, l]
        sample = sample.permute(2, 0, 1) # [l, h, w]
        sample = sample.unsqueeze(dim=0)

        return sample

    def __len__(self):
        return self.length


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Training trihit')
    # dataset and model
    parser.add_argument('--datapath', type=str, default='/home/disk1/data/')
    parser.add_argument('--output', type=str, default='./pred_adaptor_result25')
    #parser.add_argument('--output', type=str, default='./pred_result')
    parser.add_argument('--dataset', type=str, default='Indian_pines')
    parser.add_argument('--datatype', type=str, default='HSI')
    parser.add_argument('--patch_size', type=int, default=27)
    parser.add_argument('--batch_size', type=int, default=1024)#1024
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--model_name', type=str, default='trihit_cth_sdt_r5')
    parser.add_argument('--dp_rate', type=float, default=0.1)
    parser.add_argument('--sample_list', type=str, default='ab_25') #[cp, ab_25, ab_50, ab_75]
    parser.add_argument('--trained_weight_dir', type=str, default='/home/disk1/result/trihit_adapter_cifar10pretrain/Indian_pines/trihit_cth_sdt_r5_ab_25_lrs-0.25/trihit_cth_sdt_r5_best.pth')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    #parser.add_argument('--gpus', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    dataset_path = os.path.join(args.datapath, args.datatype, '{}/{}_data.npy'.format(args.dataset, args.dataset))
    test_sample_list_dir = os.path.join(args.datapath, args.datatype, '{}/sample_list/test_{}.txt'.format(args.dataset, args.sample_list))
    res_save_dir = os.path.join(args.output, args.dataset)
    if not os.path.exists(res_save_dir): os.makedirs(res_save_dir)
    res_dir = os.path.join(res_save_dir, '{}_res.npy'.format(args.sample_list))
    acc_dir = os.path.join(res_save_dir, '{}_res.text'.format(args.sample_list))
    pred_map_text = os.path.join(res_save_dir, '{}_res_map.text'.format(args.sample_list))

    if os.path.exists(res_dir):
        print('{} already exists'.format(res_dir))
        pred_map = np.load(res_dir)
    else:
        dataset = HSI(dataset_path, args.patch_size)
        val_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.num_workers)
        H, W = dataset.H, dataset.W

        model = model_builder(model_name=args.model_name, num_classes=datasets_info[args.dataset], args=args)
        model = DataParallel(model)
        model_state_dict = torch.load(args.trained_weight_dir, map_location='cpu')
        model.load_state_dict(model_state_dict)
        model = model.cuda()
        model.eval()
        # flops, params = profile(model, inputs=(torch.randn(1, 1, 200, 27, 27).cuda(),))
        # print(flops, params)

        print('start to evaluate...')

        pred_map = []
        xxx=0
        with torch.no_grad():
            for samples in tqdm(val_loader):
                samples = samples.to('cuda')
                # torch.cuda.synchronize()
                # start = time.time()
                # output = model(samples)
                # torch.cuda.synchronize()
                # end = time.time()
                # xxx = xxx + (end - start)
                # print(xxx)
                output = model(samples)
                pred_label = output.data.max(1)[1]
                pred_map.append(pred_label.cpu().numpy())

        pred_map = np.hstack(pred_map)
        with open(pred_map_text, 'w') as f:
            f.write('H:{}\n'.format(H))
            f.write('W:{}\n'.format(W))
            for pred_label in pred_map:
                f.write('{}\n'.format(pred_label))

        pred_map = np.reshape(pred_map, (H, W))
        np.save(res_dir, pred_map)

    print('start to calculate OA, AA and Kappa...')

    with open(test_sample_list_dir, 'r') as f:
        test_sample_list = f.readlines()
        test_sample_arr = [item.rstrip('\n') for item in test_sample_list[1:]]

    pred_labels = []
    gt_labels = []
    for item in test_sample_arr:
        loc_x, loc_y, label = np.array(item.split(), np.int)
        pred_labels.append(pred_map[loc_x,loc_y])
        gt_labels.append(label-1)

    pred_labels, gt_labels = np.array(pred_labels), np.array(gt_labels)
    acc_res = OA_AA_K_cal(pred_labels, gt_labels)
    with open(acc_dir, 'w') as f:
        for i, acc in enumerate(acc_res[:-3]):
            info = 'class {} || acc: {}\n'.format(i, round(acc*100, 2))
            f.write(info)

        info = 'OA: {}\n'.format(round(acc_res[-3]*100, 2))
        f.write(info)
        info = 'AA: {}\n'.format(round(acc_res[-2] * 100, 2))
        f.write(info)
        info = 'K: {}\n'.format(round(acc_res[-1] * 100, 2))
        f.write(info)

    print('done')






