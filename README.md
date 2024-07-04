# Bridging Sensor Gaps via Attention Gated Tuning for Hyperspectral Image Classification

Xizhe Xue, Haokui Zhang, Rong Xiao, Ying Li，Zongwen Bai, Mike Zheng Shou

[[`arXiv`](https://arxiv.org/abs/2309.12865)] [[`BibTeX`](#CitingAGT)]

<div align="center"

  ![graphic-abstract](https://cdn.statically.io/gh/Cecilia-xue/picx-images-hosting@master/graphic-abstract.2doko6fnqq.webp)
  
</div><br/>

## Features

* Attention-Gated tuning approach for HSI Classification.
* Highly efficient triplet-structured HSI classification transformer 
* Support cross-sensor or cross-modality tuning, allowing for the utilization of abundant RGB labeled datasets to enhance HSI classification performance


## Datasets preparing

The current version AGT has support for a few datasets. Due to Policy constraints, we are not able to directly provide and host HSI images. However, we share the pre-processed HSI images in .h5 and .mat files. Datasets can be downloaded by accessing [Google Drive](https://drive.google.com/drive/folders/1DM_I__KRbyzV88De8Y4lL8k4VDPYgTTz?usp=sharing).

The data preprocessing files are in the *data* directory. Running these scripts can complete the data preprocessing.
## Tri-Former Model training

#### Indian pines
- `python train_dist.py --dataset Indian_pines --model_name trihit_cth --sample_list cp --batch_size 12 --epochs 300`


#### Pavia University
- `python train_dist.py --dataset PaviaU --model_name trihit_cth --sample_list cp --batch_size 12 --epochs 300`

#### Pavia Center
- `python train_dist.py --dataset PaviaC --model_name trihit_cth --sample_list cp --batch_size 12  --epochs 300`

#### houstonu
- `python train_dist.py --dataset HoustonU --model_name trihit_cth --sample_list cp --batch_size 12 --epochs 300`

#### Salinas
- `python train_dist.py --dataset Salinas --model_name trihit_cth --sample_list cp --batch_size 12 --epochs 300`

#### Cifar
- `python train_dist.py --dataset cifar10 --datatype PHSI --model_name trihit_cth --sample_list cp --batch_size 12 --epochs 150`
- `python train_dist.py --dataset cifar100 --datatype PHSI --model_name trihit_cth --sample_list cp --batch_size 12 --epochs 150`

### Inference Tri-Form Models
The pre-trained model can be found in <a href="https://drive.google.com/drive/folders/172unB7hKEKn2gdMd75ytj6ICQ7gONX4q?usp=sharing">Tri-Former model</a>.

## AGT Tuning

#### Salinas 2 Indian pines

- `python train_adapter.py --dataset Indian_pines --freeze_model_dir '/home/disk1/result/trihit/Salinas/trihit_cth_cp/trihit_cth_best.pth' --model_name trihit_cth_sdt_r5 --sample_list ab_50 --batch_size 24  --epochs 150`

#### Salinas 2 HoustonU
- `python train_adapter.py --dataset HoustonU --freeze_model_dir '/home/disk1/result/trihit/Salinas/trihit_cth_cp/trihit_cth_best.pth' --model_name trihit_cth_sdt_r5  --sample_list ab_50 --batch_size 24  --epochs 150`

##### Salinas 2 PaviaU
- `python train_adapter.py --dataset PaviaU --freeze_model_dir '/home/disk1/result/trihit/Salinas/trihit_cth_cp/trihit_cth_best.pth' --model_name trihit_cth_sdt_r5  --sample_list ab_50 --batch_size 24  --epochs 150`

##### PaviaC 2 indian pines
- `python train_adapter.py --dataset Indian_pines --freeze_model_dir '/home/disk1/result/trihit/PaviaC/trihit_cth_cp/trihit_cth_best.pth' --model_name trihit_cth_sdt_r5  --sample_list ab_50 --batch_size 24  --epochs 150`

##### PaviaC 2 HoustonU
- `python train_adapter.py --dataset HoustonU --freeze_model_dir '/home/disk1/result/trihit/PaviaC/trihit_cth_cp/trihit_cth_best.pth' --model_name trihit_cth_sdt_r5  --sample_list ab_50 --batch_size 24  --epochs 150`

#### PaviaC 2 PaviaU
- `python train_adapter.py --dataset PaviaU --freeze_model_dir '/home/disk1/result/trihit/PaviaC/trihit_cth_cp/trihit_cth_best.pth' --model_name trihit_cth_sdt_r5  --sample_list ab_50 --batch_size 24  --epochs 150`

#### Cifar 2 Indian pines
- `python train_adapter.py --dataset Indian_pines --freeze_model_dir '/home/disk1/result/trihit/cifar/trihit_cth_cp/trihit_cth_best.pth' --model_name trihit_cth_sdt_r5  --sample_list ab_50 --batch_size 24  --epochs 150`

#### Cifar 2 HoustonU
- `python train_adapter.py --dataset HoustonU --freeze_model_dir '/home/disk1/result/trihit/cifar/trihit_cth_cp/trihit_cth_best.pth' --model_name trihit_cth_sdt_r5  --sample_list ab_50 --batch_size 24  --epochs 150`

#### Cifar 2 PaviaU
- `python train_adapter.py --dataset PaviaU --freeze_model_dir '/home/disk1/result/trihit/cifar/trihit_cth_cp/trihit_cth_best.pth' --model_name trihit_cth_sdt_r5  --sample_list ab_50 --batch_size 24  --epochs 150`

### Inference with Models
Please edit *eval_dp.py*, enter the model path and test data set in the parameter section, and run eval.py to test the model results.




Pick the fintuned-model from [model zoo](MODEL_ZOO.md). More models trained with different samples are coming !
## Model Zoo
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Dataset</th>
<th valign="bottom">Number of training samples</th>
<th valign="bottom">OA%</th>
<th valign="bottom">AA%</th>
<th valign="bottom">K%</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_swin_small_bs16_50ep -->
 <tr><td align="center">AGT+Tri-Former</td>
<td align="center">Salinas 2 Indian Pines</td>
<td align="center">50 pixel/class</td>
<td align="center">94.83</td>
<td align="center">97.62</td>
<td align="center">94.07</td>
<td align="center"><a href="https://drive.google.com/file/d/19dU3Uwfcfvt2N4nIqiv1xPmWdvcqqq4o/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_384_bs16_50ep -->
 <tr><td align="center">AGT+Tri-Former</td>
<td align="center">Salinas 2 Pavia University</td>
<td align="center">50 pixel/class</td>
<td align="center">98.51</td>
<td align="center">98.40</td>
<td align="center">98.22</td>
<td align="center"><a href="https://drive.google.com/file/d/1TJVMu6FbJzqypRdnVsd7m2klwxe3M6hY/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_50ep -->
 <tr><td align="center">AGT+Tri-Former</td>
<td align="center">Salinas 2 Houston University</td>
<td align="center">50 pixel/class</td>
<td align="center">92.65</td>
<td align="center">93.86</td>
<td align="center">92.05</td>
<td align="center"><a href="https://drive.google.com/file/d/1-iy1KUnz0VxN_Oqj5Vek00tgnEQv2iJV/view?usp=sharing">model</a></td>
</tr>
</tbody></table>

## <a name="Citing AGT"></a>Citing HyT-NAS

If you use AGT in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry. 

```BibTeX
@misc{xue2023bridgingsensorgapssingledirection,
      title={Bridging Sensor Gaps via Attention Gated Tuning for Hyperspectral Image Classification}, 
      author={Xizhe Xue and Haokui Zhang and Rong Xiao and Ying Li and Zongwen Bai and Mike Zheng Shou},
      year={2023},
      eprint={2309.12865},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2309.12865}, 
}
```

If you have any questions, please feel free to contact me via <a href="xuexizhe@mail.nwpu.edu.cn">e-mail</a> . 