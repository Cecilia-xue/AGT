import os
import argparse
import torch.optim as optim
from torch.nn import DataParallel
from engine import *
from datasets_info import datasets_info
from data.data_builder import data_builder
from models.model_builder import model_builder
from timm.loss import LabelSmoothingCrossEntropy
from utils import TxtLogger, cosine_scheduler, resume_dir_get
import torch.distributed as dist
import torch.multiprocessing as mp
# from glob import glob
# from thop import profile
# from torchstat import stat
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
warnings.filterwarnings("ignore")
def train(rank, world_size, args, log_writer):
    # Initialize the distributed training environment
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # build dataloader
    train_set, val_set, num_training_steps_per_epoch = data_builder(args)
    train_sampler = torch.utils.data.DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    # Create the network model
    model = model_builder(model_name=args.model_name, num_classes=datasets_info[args.dataset], args=args)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # optimizer, lr_sheduler, criterion
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs=args.warmup_epochs
    )
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    # resume
    start_epoch = args.start_epoch[0] if isinstance(args.start_epoch, tuple) else args.start_epoch
    if not start_epoch==0:
        print('resume model from {}'.format(args.resume_dir))
        model_state_dict = torch.load(args.resume_dir, map_location=torch.device('cuda', rank))
        model.load_state_dict(model_state_dict)

    output_dir = os.path.join(args.output, args.dataset, '{}_{}'.format(args.model_name, args.sample_list))

    # Training loop
    if rank==0:
        best_val_acc=0
    print('training start from epoch {} '.format(start_epoch+1))
    for epoch in range(start_epoch+1, args.epochs):
        train_sampler.set_epoch(epoch)
        model, train_loss, train_acc = train_one_epoch(model, criterion, lr_schedule_values,
                                                       start_steps=epoch*num_training_steps_per_epoch,
                                                       train_loader=train_loader, optimizer=optimizer, device=rank)

        if rank==0:
            log_info = 'rank {} epoch:{} || ema train loss:{} train acc:{}'.format(rank, epoch, train_loss, train_acc)
            print(log_info)
            log_writer.update(log_info)
            torch.save(model.state_dict(), os.path.join(output_dir, '{}_{}.pth'.format(args.model_name, epoch)))
            old_pth_dir = os.path.join(output_dir, '{}_{}.pth'.format(args.model_name, epoch-20))
            if os.path.exists(old_pth_dir):
                os.remove(old_pth_dir)
            if epoch % args.eval_interval == 0:
                sample_num, avg_loss, acc = evaluate(model, criterion, val_loader, device=rank)
                if acc > best_val_acc:
                    best_val_acc=acc
                    torch.save(model.state_dict(),
                               os.path.join(output_dir, '{}_best.pth'.format(args.model_name)))
                log_info = '{} samples epoch: {} || val loss:{} val acc:{} best val acc:{}'.format(sample_num, epoch, avg_loss, acc, best_val_acc)
                print(log_info)
                log_writer.update(log_info)

    # Clean up
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Training trihit')
    # dataset and model
    parser.add_argument('--root', type=str, default='/home/disk1/data')
    parser.add_argument('--output', type=str, default='/home/disk1/result/trihit')
    parser.add_argument('--dataset', type=str, default='Indian_pines')
    parser.add_argument('--datatype', type=str, default='HSI')
    parser.add_argument('--patch_size', type=int, default=27)
    parser.add_argument('--model_name', type=str, default='trihit_cth')
    parser.add_argument('--sample_list', type=str, default='cp') #[cp, ab_25, ab_50, ab_75]
    parser.add_argument('--resume', type=bool, default=True)

    # optimizer
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dp_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=304)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--seed', default=1)
    parser.add_argument('--eval_interval', type=int, default=5)

    # distributed training
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpus', default='3')
    #parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--port', type=int, default=23457)

    # augmentation
    parser.add_argument('--flip', type=float, default=0.5)
    parser.add_argument('--rotate', type=float, default=0.5)
    parser.add_argument('--add_noise', type=float, default=None)
    parser.add_argument('--cutout', type=float, nargs='+', default=[0.3, 0.3, 0.1])
    parser.add_argument('--rcutout', type=float, nargs='+', default=[0.0, 0.0, 0.0])

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    output_dir = os.path.join(args.output, args.dataset, '{}_{}'.format(args.model_name, args.sample_list))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_gpus = torch.cuda.device_count()

    if args.resume:
        start_epoch, resume_dir = resume_dir_get(args)
        args.start_epoch = start_epoch,
        args.resume_dir = resume_dir
    else:
        args.start_epoch=0
    log_txt = os.path.join(output_dir, '{}_log.txt'.format(args.model_name))
    log_writer = TxtLogger(log_dir=log_txt)

    mp.spawn(train, args=(num_gpus, args, log_writer), nprocs=num_gpus)