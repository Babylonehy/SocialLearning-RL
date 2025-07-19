import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from train_loops import test
from torch.utils.tensorboard import SummaryWriter
from models import model_dict
# from setting import  teacher_model_path_dict  # 注释掉teacher相关
from dataset.cifar100 import get_cifar100_dataloaders
from utils import set_logger
# from models.util import Regress, TransFeat  # 注释掉teacher相关
import torch.nn.functional as F
from distiller_zoo import FeatureKLLoss, FeatureMSELoss

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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--ce-weight', type=float, default=1, help='ce loss coefficient')
parser.add_argument('--kd-weight', type=float, default=1, help='kd loss coefficient')
parser.add_argument('--feat-weight', type=float, default=5, help='kd loss coefficient')
parser.add_argument('--milestones', default=[150,180,210], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--feat-kd', default='mse', type=str, help='feature kd loss')
parser.add_argument('--kd-T', type=int, default=4, help='temperature')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')
parser.add_argument('--student-name-list', default=['resnet32x4', 'wrn_28_4'], type=str, nargs='+', help='student models')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'tinyimagenet', 'dogs', 'cub_200_2011', 'mit67'], help='dataset')
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--warmup-epochs', type=int, default=10, help='number of warmup epochs')


def train_kd(train_loader, student, teacher, criterion_ce, criterion_kd, optimizer, epoch, device, args):
    student.train()
    teacher.eval()
    if args.feat_kd == 'mse':
        feat_kd_func = FeatureMSELoss()
    elif args.feat_kd == 'kl':
        feat_kd_func = FeatureKLLoss(args.kd_T)
    else:
        feat_kd_func = None
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        s_feats, s_logits = student(inputs, is_feat=True)
        with torch.no_grad():
            t_feats, t_logits = teacher(inputs, is_feat=True)
        loss_ce = criterion_ce(s_logits, targets)
        loss_kd = criterion_kd(s_logits, t_logits)
        if feat_kd_func is not None:
            loss_feat = feat_kd_func(s_feats[-2], t_feats[-2])
        else:
            loss_feat = 0.
        loss = args.ce_weight * loss_ce + args.kd_weight * loss_kd + args.feat_weight * loss_feat
        loss.backward()
        optimizer.step()


def main():
    args = parser.parse_args()
    args.student_name_str = "_".join(args.student_name_list)
    print('args.student_name_str', args.student_name_str)
    args.student_num = len(args.student_name_list)

    args.model_name = args.arch + '_'+ args.dataset+ '_'+ 'rl'+'_'+ args.trial+'_'+str(args.student_num)+'_'+args.student_name_str

    info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    info = args.model_name + info_time
    print(f'===> info is : {info}')
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    print(f'===>args.checkpoint_dir is : {args.checkpoint_dir}')

    args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')
    args.logger = set_logger(args.log_txt)
    args.logger.info("==========\nArgs:{}\n==========".format(args))

    if args.seed is not None :
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 数据加载
    if args.dataset.startswith('cifar100'):
        args.n_cls = 100
        args.res = (1, 3, 32, 32)
    elif args.dataset.startswith('imagenet'):
        args.n_cls = 1000
        args.res = (1, 3, 224, 224)
    train_loader, val_loader = get_cifar100_dataloaders(data_folder=args.data,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.workers)

    # 初始化多个学生模型
    student_models = []
    student_optimizers = []
    for name in args.student_name_list:
        model = model_dict[name](num_classes=args.n_cls).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        student_models.append(model)
        student_optimizers.append(optimizer)

    criterion_ce = nn.CrossEntropyLoss().to(device)
    from utils import DistillKL
    criterion_div = DistillKL(args.kd_T).to(device)

    best_accs = [0. for _ in range(args.student_num)]

    # warmup阶段
    for epoch in range(args.warmup_epochs):
        for idx, (model, optimizer) in enumerate(zip(student_models, student_optimizers)):
            # 只用CE loss
            model.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion_ce(outputs, targets)
                loss.backward()
                optimizer.step()
            acc = test(epoch, model, device, val_loader, criterion_ce, args)
            if acc > best_accs[idx]:
                best_accs[idx] = acc
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'student{idx}_best.pth'))
            args.logger.info(f'Warmup Epoch {epoch} Student {idx} Acc: {acc:.2f}')

    # 轮流当老师阶段
    for epoch in range(args.warmup_epochs, args.epochs):
        teacher_idx = epoch % args.student_num
        teacher_model = student_models[teacher_idx]
        for idx, (model, optimizer) in enumerate(zip(student_models, student_optimizers)):
            if idx == teacher_idx:
                continue
            train_kd(train_loader, model, teacher_model, criterion_ce, criterion_div, optimizer, epoch, device, args)
            acc = test(epoch, model, device, val_loader, criterion_ce, args)
            if acc > best_accs[idx]:
                best_accs[idx] = acc
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'student{idx}_best.pth'))
            args.logger.info(f'Epoch {epoch} Student {idx} Acc: {acc:.2f}')

    # 评估所有学生
    for idx, model in enumerate(student_models):
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f'student{idx}_best.pth')))
        acc = test(args.epochs, model, device, val_loader, criterion_ce, args)
        args.logger.info(f'Final Test Student {idx} Best Acc: {acc:.2f}')

if __name__ == '__main__' :
     main() 
