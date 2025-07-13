import argparse
import os
import random
import shutil
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.imagenet import get_imagenet_dataloaders
from utils import set_logger, DistillKL
from train_loops_multi import train_single_student, train_student_with_kd

def main():
    parser = argparse.ArgumentParser(description='Mutual Learning: Multi-Student, No Teacher')
    parser.add_argument('--data', metavar='DIR', default='imagenet', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'])
    parser.add_argument('--student-archs', default=['resnet18_imagenet'], type=str, nargs='+', help='list of student architectures')
    parser.add_argument('--epochs', default=240, type=int, help='total epochs')
    parser.add_argument('--warmup-ratio', default=0.1, type=float, help='warmup ratio (0~1)')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--kd-T', type=int, default=4, help='KD temperature')
    parser.add_argument('--kd-weight', type=float, default=1.0)
    parser.add_argument('--ce-weight', type=float, default=1.0)
    parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str)
    parser.add_argument('--trial', type=str, default='1')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    # Logger
    info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    model_name = "_".join(args.student_archs) + f"_{args.dataset}_mutual_{args.trial}" + info_time
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_txt = os.path.join(args.checkpoint_dir, model_name + '.txt')
    args.logger = set_logger(log_txt)
    args.logger.info("==========\nArgs:{}\n==========".format(args))

    # Data
    if args.dataset == 'cifar100':
        args.n_cls = 100
        args.res = (1, 3, 32, 32)
        train_loader, val_loader = get_cifar100_dataloaders(data_folder=args.data, batch_size=args.batch_size, num_workers=args.workers)
    elif args.dataset == 'imagenet':
        args.n_cls = 1000
        args.res = (1, 3, 224, 224)
        train_loader, val_loader = get_imagenet_dataloaders(data_folder=args.data, batch_size=args.batch_size, num_workers=args.workers)
    else:
        raise ValueError('Unsupported dataset')

    # Models
    student_models = []
    optimizers = []
    for arch in args.student_archs:
        model = model_dict[arch](num_classes=args.n_cls).cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        student_models.append(model)
        optimizers.append(optimizer)

    criterion_ce = nn.CrossEntropyLoss().cuda()
    criterion_kd = DistillKL(args.kd_T).cuda()

    warmup_epochs = int(args.epochs * args.warmup_ratio)
    num_students = len(student_models)
    best_accs = [0.] * num_students

    for epoch in range(args.epochs):
        if epoch < warmup_epochs:
            # Warmup: each student trains independently
            for s_idx, (model, optimizer) in enumerate(zip(student_models, optimizers)):
                train_single_student(train_loader, model, criterion_ce, optimizer, epoch, torch.device('cuda'), args)
        else:
            # Mutual learning: rotate teacher
            teacher_idx = epoch % num_students
            teacher_model = student_models[teacher_idx].eval()
            for s_idx, (model, optimizer) in enumerate(zip(student_models, optimizers)):
                if s_idx == teacher_idx:
                    train_single_student(train_loader, model, criterion_ce, optimizer, epoch, torch.device('cuda'), args)
                else:
                    train_student_with_kd(train_loader, model, teacher_model, criterion_ce, criterion_kd, optimizer, epoch, torch.device('cuda'), args, kd_T=args.kd_T, kd_weight=args.kd_weight, ce_weight=args.ce_weight)
        # TODO: 验证与保存模型

if __name__ == '__main__':
    main()