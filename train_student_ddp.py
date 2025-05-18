import argparse
import os
import random
import shutil
import time
import warnings
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models import model_dict
from setting import teacher_model_path_dict
from dataset.cifar100 import get_cifar100_dataloaders
from utils import set_logger, cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num
from models.util import TransFeat

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DDP Training')
    parser.add_argument('--data', metavar='DIR', default='./data',
                    help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
    parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use')
    parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training')
    parser.add_argument('--kd-T', type=int, default=4, help='temperature')
    parser.add_argument('--feat-kd', default='mse', type=str, help='feature kd loss')
    parser.add_argument('--feat-weight', type=float, default=5, help='feature loss weight')
    parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')
    parser.add_argument('--teacher-name-list', default=['resnet32x4', 'wrn_28_4'], type=str, nargs='+', help='teacher models')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'], help='dataset')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    return parser.parse_args()

class DistillationModel(nn.Module):
    def __init__(self, student_model, teacher_models, feat_trans):
        super(DistillationModel, self).__init__()
        self.student = student_model
        self.teachers = nn.ModuleList(teacher_models)
        self.feat_trans = feat_trans
        
    def forward(self, x, is_feat=False):
        student_features, student_logits = self.student(x, is_feat=True)
        trans_student_features = self.feat_trans(student_features[-2])
        
        teacher_logits = []
        teacher_features = []
        with torch.no_grad():
            for teacher in self.teachers:
                t_features, t_logits = teacher(x, is_feat=True)
                teacher_features.append(t_features[-2])
                teacher_logits.append(t_logits)
                
        if is_feat:
            return student_features, student_logits, trans_student_features, teacher_features, teacher_logits
        return student_logits

def load_teacher(model_path, n_cls, model_t, gpu):
    model = model_dict[model_t](num_classes=n_cls).cuda(gpu)
    model.load_state_dict(torch.load(model_path, map_location=f'cuda:{gpu}')['model'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_feat_trans(model, args, gpu):
    model.eval()
    with torch.no_grad():
        s_feat, s_logits = model(torch.rand(args.res).cuda(gpu), is_feat=True)
    args.s_feat_dim = s_feat[-2].size()
    
    # Get teacher feature dimensions
    teacher_models = [load_teacher(teacher_model_path_dict[name], args.n_cls, name, gpu) 
                     for name in args.teacher_name_list]
    x = torch.rand(args.res).cuda(gpu)
    feature_dims = []
    for t in teacher_models:
        feature, _ = t(x, is_feat=True)
        feature_dims.append(feature[-2].size())
    
    return TransFeat(args.s_feat_dim, feature_dims).cuda(gpu)

def train(train_loader, model, criterion_list, optimizer, epoch, device, args):
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_kd = AverageMeter('train_loss_kd', ':.4e')
    train_loss_feat = AverageMeter('train_loss_feat', ':.4e')

    top1_num = 0
    top5_num = 0
    total = 0

    lr = adjust_lr(optimizer, epoch, args)
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]

    model.train()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_start_time = time.time()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        student_features, student_logits, trans_student_features, teacher_features, teacher_logits = model(inputs, is_feat=True)
        
        loss_cls = criterion_ce(student_logits, targets)
        
        loss_kd = torch.tensor(0.).cuda(args.gpu)
        for idx in range(len(model.teachers)):
            loss_kd = loss_kd + criterion_div(student_logits, teacher_logits[idx])
        loss_kd = loss_kd / len(model.teachers)
        
        loss_feat = torch.tensor(0.).cuda(args.gpu)
        if args.feat_kd == 'mse':
            feat_kd_func = nn.MSELoss()
        elif args.feat_kd == 'kl':
            feat_kd_func = nn.KLDivLoss()

        for idx in range(len(model.teachers)):
            loss_feat = loss_feat + feat_kd_func(trans_student_features[idx], teacher_features[idx])
        loss_feat = loss_feat / len(model.teachers)
        loss_feat = args.feat_weight * loss_feat
        
        loss = loss_cls + loss_kd + loss_feat
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))
        train_loss_kd.update(loss_kd.item(), inputs.size(0))
        train_loss_feat.update(loss_feat.item(), inputs.size(0))
        
        top1, top5 = correct_num(student_logits, targets, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += targets.size(0)

        if args.rank == 0 and batch_idx % args.print_freq == 0:
            print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, CLS Loss:{:.2f},' 
                'KD Loss:{:.2f}, Feature Loss:{:.2f}, Top-1 Acc:{:.2f}'.format(
                epoch, batch_idx, len(train_loader), lr, time.time()-batch_start_time, 
                train_loss_cls.avg, train_loss_kd.avg, train_loss_feat.avg, 
                (top1_num/total*100.).item()))
    
    return top1_num / total

def test(val_loader, model, criterion, device, args):
    model.eval()
    test_loss = AverageMeter('test_loss', ':.4e')
    top1_num = 0
    top5_num = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            student_logits = model(inputs)
            loss = criterion(student_logits, targets)
            
            test_loss.update(loss.item(), inputs.size(0))
            top1, top5 = correct_num(student_logits, targets, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += targets.size(0)
            
    if args.rank == 0:
        print('Test Loss: {:.4f}, Top-1 Acc: {:.2f}%, Top-5 Acc: {:.2f}%'.format(
            test_loss.avg, (top1_num/total*100.).item(), (top5_num/total*100.).item()))
    
    return top1_num / total

def main():
    args = parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')
    
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                              world_size=args.world_size, rank=args.rank)
    
    # Setup dataset
    if args.dataset == 'cifar100':
        args.n_cls = 100
        args.res = (1, 3, 32, 32)
    elif args.dataset == 'imagenet':
        args.n_cls = 1000
        args.res = (1, 3, 224, 224)
    
    # Load student model
    student_model = model_dict[args.arch](num_classes=args.n_cls).cuda(args.gpu)
    
    # Load teacher models
    teacher_models = [load_teacher(teacher_model_path_dict[name], args.n_cls, name, args.gpu) 
                     for name in args.teacher_name_list]
    
    # Get feature transformer
    feat_trans = get_feat_trans(student_model, args, args.gpu)
    
    # Create combined model
    model = DistillationModel(student_model, teacher_models, feat_trans)
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = DDP(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = DDP(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    # Setup loss functions and optimizer
    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_div = DistillKL(args.kd_T).cuda(args.gpu)
    criterion_list = nn.ModuleList([criterion_ce, criterion_div])
    
    optimizer = optim.SGD(model.parameters(),
                         lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)
    
    # Load data
    train_loader, val_loader = get_cifar100_dataloaders(
        data_folder=args.data,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_acc = train(train_loader, model, criterion_list, optimizer, epoch, args.gpu, args)
        
        # Evaluate on validation set
        val_acc = test(val_loader, model, criterion_ce, args.gpu, args)
        
        # Save checkpoint
        if args.rank == 0:
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            
            # Only save student model weights
            student_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'model': model.module.student.state_dict() if args.distributed else model.student.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            
            save_path = os.path.join(args.checkpoint_dir, f'{args.arch}_checkpoint.pth.tar')
            torch.save(student_state, save_path)
            if is_best:
                shutil.copyfile(save_path, os.path.join(args.checkpoint_dir, f'{args.arch}_best.pth.tar'))

if __name__ == '__main__':
    main() 
