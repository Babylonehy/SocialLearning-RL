import argparse
import os
import random
import shutil
import time
import warnings
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from train_loops import train, test
from models import model_dict
from setting import teacher_model_path_dict
from dataset.cifar100 import get_cifar100_dataloaders
from utils import set_logger, DistillKL
from models.util import TransFeat

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_imagenet')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
    parser.add_argument('--dynamic', action='store_true', help="use dynamic weight aggregation strategy")
    parser.add_argument('--ce-weight', type=float, default=1, help='ce loss coefficient')
    parser.add_argument('--kd-weight', type=float, default=1, help='kd loss coefficient')
    parser.add_argument('--feat-weight', type=float, default=5, help='kd loss coefficient')
    parser.add_argument('--milestones', default=[150,180,210], type=int, nargs='+', help='milestones for lr-multistep')
    parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
    parser.add_argument('--feat-kd', default='mse', type=str, help='feature kd loss')
    parser.add_argument('--kd-T', type=int, default=4, help='temperature')
    parser.add_argument('--agent-step', type=int, default=1000, help='agent optimization step')
    parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')
    parser.add_argument('--teacher-name-list', default=['resnet32x4', 'wrn_28_4'], type=str, nargs='+', help='teacher models')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'tinyimagenet', 'dogs', 'cub_200_2011', 'mit67'], help='dataset')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    return parser.parse_args()

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://',
                          world_size=world_size, rank=rank)
    return rank, world_size, gpu

def load_teacher(model_path, n_cls, model_t, gpu):
    model = model_dict[model_t](num_classes=n_cls).cuda(gpu)
    model.load_state_dict(torch.load(model_path, map_location=f'cuda:{gpu}')['model'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_agent(teacher_models, args, gpu):
    teacher_num = len(teacher_models)
    x = torch.rand(args.res).cuda(gpu)
    logits_dim = 0
    feature_dim = 0
    feature_dims = []
    policy_input_size = []
    
    for t in teacher_models:
        feature, logits = t(x, is_feat=True)
        logits_dim += logits.size(1)
        feature_dim += feature[-1].size(1)
        feature_dims.append(feature[-2].size())
        policy_input_size.append(feature[-1].size(1) + logits.size(1) + 3)
        
    agent = model_dict['PolicyTrans'](policy_input_size, teacher_num, args.dynamic).cuda(gpu)
    return agent, feature_dims

def main():
    args = parse_args()
    
    # Setup distributed training
    rank, world_size, gpu = setup_distributed()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup logging
    if rank == 0:
        args.teacher_name_str = "_".join(args.teacher_name_list)
        args.model_name = f"{args.arch}_{args.dataset}_rl_{args.trial}_{len(args.teacher_name_list)}_{args.teacher_name_str}"
        info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        info = args.model_name + info_time
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')
        args.logger = set_logger(args.log_txt)
        args.logger.info(f"Args: {args}")
    
    # Setup dataset
    if args.dataset == 'cifar100':
        args.n_cls = 100
        args.res = (1, 3, 32, 32)
    elif args.dataset == 'imagenet':
        args.n_cls = 1000
        args.res = (1, 3, 224, 224)
    
    # Load teachers
    teacher_models = [load_teacher(teacher_model_path_dict[name], args.n_cls, name, gpu) 
                     for name in args.teacher_name_list]
    
    # Load student model
    model = model_dict[args.arch](num_classes=args.n_cls).cuda(gpu)
    model = DDP(model, device_ids=[gpu])
    
    # Setup agent and feature transformer
    agent, args.t_feat_dims = get_agent(teacher_models, args, gpu)
    agent = DDP(agent, device_ids=[gpu])
    
    feat_trans = TransFeat(args.s_feat_dim, args.t_feat_dims).cuda(gpu)
    feat_trans = DDP(feat_trans, device_ids=[gpu])
    
    # Setup loss functions
    criterion_ce = nn.CrossEntropyLoss().cuda(gpu)
    criterion_div = DistillKL(args.kd_T).cuda(gpu)
    criterion_list = nn.ModuleList([criterion_ce, criterion_div])
    
    # Setup optimizers
    trainable_list = nn.ModuleList([model, feat_trans])
    optimizer = optim.SGD(trainable_list.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    agent_optimizer = optim.SGD(agent.parameters(), lr=args.lr)
    
    # Load data
    train_loader, val_loader = get_cifar100_dataloaders(
        data_folder=args.data,
        batch_size=args.batch_size,
        num_workers=4,
        distributed=True
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        train(train_loader, model, criterion_list, optimizer, epoch, gpu, args, 
              agent, feat_trans, teacher_models, agent_optimizer)
        
        acc = test(epoch, model, gpu, val_loader, criterion_ce, args)
        
        if rank == 0:
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'model': model.module.state_dict(),
                'acc': acc,
                'optimizer': optimizer.state_dict()
            }
            
            torch.save(state, os.path.join(args.checkpoint_dir, f'{args.arch}.pth.tar'))
            
            if acc > best_acc:
                best_acc = acc
                shutil.copyfile(
                    os.path.join(args.checkpoint_dir, f'{args.arch}.pth.tar'),
                    os.path.join(args.checkpoint_dir, f'{args.arch}_best.pth.tar')
                )
    
    if rank == 0:
        args.logger.info(f'Best accuracy: {best_acc}')

if __name__ == '__main__':
    main() 
