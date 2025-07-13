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
from dataset.cifiar100_subset import get_cifar100_dataloaders_subset
from utils import set_logger, DistillKL
from train_loops_multi import train_single_student, train_student_with_kd
from helper.loops import validate
from helper.util import save_dict_to_json

def parse_student_subsets(student_archs, student_class_lists):
    """
    student_class_lists: 逗号分隔的字符串列表，每个学生一个子集，如 '0,1,2;3,4,5;6,7,8'
    返回: list of list of int
    """
    if student_class_lists is None:
        return [None for _ in student_archs]
    split_lists = student_class_lists.split(';')
    assert len(split_lists) == len(student_archs), 'class_list数目需与学生模型数一致'
    result = []
    for s in split_lists:
        if s.strip() == '' or s.strip().lower() == 'none':
            result.append(None)
        else:
            result.append([int(x) for x in s.split(',')])
    return result

def main():
    parser = argparse.ArgumentParser(description='Mutual Learning: Multi-Student, No Teacher')
    parser.add_argument('--data', metavar='DIR', default='imagenet', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'])
    parser.add_argument('--student-archs', default=['resnet18_imagenet'], type=str, nargs='+', help='list of student architectures')
    parser.add_argument('--student-class-lists', type=str, default=None, help='每个学生的子集类别,如 "0,1,2;3,4,5;6,7,8"')
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
    parser.add_argument('--print-freq', type=int, default=100)
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

    # 解析每个学生的类别子集
    student_class_lists = parse_student_subsets(args.student_archs, args.student_class_lists)

    # Data & Model
    student_models = []
    optimizers = []
    train_loaders = []
    val_loaders = []
    best_accs = []
    save_folders = []
    for idx, (arch, class_list) in enumerate(zip(args.student_archs, student_class_lists)):
        # 数据加载
        if args.dataset == 'cifar100':
            if class_list is not None:
                train_loader, val_loader, _, _ = get_cifar100_dataloaders_subset(
                    class_list=class_list,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    root=args.data
                )
            else:
                train_loader, val_loader = get_cifar100_dataloaders(
                    args.data, batch_size=args.batch_size, num_workers=args.workers)
            n_cls = 100
        else:
            raise NotImplementedError('Only cifar100 with subset is supported in this demo')
        # 模型
        model = model_dict[arch](num_classes=n_cls).cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        # 保存路径
        if class_list is not None:
            sub_name = f"{arch}_subset_{'-'.join(map(str, class_list))}_trial_{args.trial}"
        else:
            sub_name = f"{arch}_full_trial_{args.trial}"
        save_folder = os.path.join(args.checkpoint_dir, sub_name)
        os.makedirs(save_folder, exist_ok=True)
        # 记录
        student_models.append(model)
        optimizers.append(optimizer)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        best_accs.append(0.)
        save_folders.append(save_folder)

    criterion_ce = nn.CrossEntropyLoss().cuda()
    criterion_kd = DistillKL(args.kd_T).cuda()
    warmup_epochs = int(args.epochs * args.warmup_ratio)
    num_students = len(student_models)

    # 加载完整测试集
    if args.dataset == 'cifar100':
        _, full_val_loader = get_cifar100_dataloaders(
            args.data, batch_size=args.batch_size, num_workers=args.workers)
    else:
        raise NotImplementedError('Only cifar100 with subset is supported in this demo')
    best_full_accs = [0.] * num_students

    for epoch in range(1, args.epochs + 1):
        # 训练
        if epoch <= warmup_epochs:
            for s_idx, (model, optimizer, train_loader) in enumerate(zip(student_models, optimizers, train_loaders)):
                train_single_student(train_loader, model, criterion_ce, optimizer, epoch, torch.device('cuda'), args)
        else:
            teacher_idx = (epoch - warmup_epochs - 1) % num_students
            teacher_model = student_models[teacher_idx].eval()
            for s_idx, (model, optimizer, train_loader) in enumerate(zip(student_models, optimizers, train_loaders)):
                if s_idx == teacher_idx:
                    train_single_student(train_loader, model, criterion_ce, optimizer, epoch, torch.device('cuda'), args)
                else:
                    train_student_with_kd(train_loader, model, teacher_model, criterion_ce, criterion_kd, optimizer, epoch, torch.device('cuda'), args, kd_T=args.kd_T, kd_weight=args.kd_weight, ce_weight=args.ce_weight)
        # 验证与保存
        for s_idx, (model, val_loader, save_folder) in enumerate(zip(student_models, val_loaders, save_folders)):
            acc, acc_top5, val_loss = validate(val_loader, model, criterion_ce, args)
            args.logger.info(f"Epoch {epoch} Student{s_idx} [{args.student_archs[s_idx]}] val_acc {acc:.3f} val_loss {val_loss:.4f}")
            # 保存最优模型（子集）
            if acc > best_accs[s_idx]:
                best_accs[s_idx] = acc
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc': best_accs[s_idx],
                    'optimizer': optimizers[s_idx].state_dict(),
                }
                torch.save(state, os.path.join(save_folder, f"{args.student_archs[s_idx]}_best.pth"))
                save_dict_to_json({
                    'val_loss': float(val_loss),
                    'val_acc': float(acc),
                    'epoch': epoch
                }, os.path.join(save_folder, "val_best_metrics.json"))
                args.logger.info(f"Saved best model for student{s_idx} (by acc)")
            # 在完整测试集上评估
            full_acc, full_acc_top5, full_val_loss = validate(full_val_loader, model, criterion_ce, args)
            args.logger.info(f"Epoch {epoch} Student{s_idx} [{args.student_archs[s_idx]}] [Full] val_acc {full_acc:.3f} val_loss {full_val_loss:.4f}")
            # 保存最优模型（全体测试集）
            if full_acc > best_full_accs[s_idx]:
                best_full_accs[s_idx] = full_acc
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_full_acc': best_full_accs[s_idx],
                    'optimizer': optimizers[s_idx].state_dict(),
                }
                torch.save(state, os.path.join(save_folder, f"{args.student_archs[s_idx]}_best_full.pth"))
                save_dict_to_json({
                    'val_loss': float(full_val_loss),
                    'val_acc': float(full_acc),
                    'epoch': epoch
                }, os.path.join(save_folder, "val_best_metrics_full.json"))
                args.logger.info(f"Saved best model for student{s_idx} (by full acc)")
    for s_idx, (best_acc, best_full_acc) in enumerate(zip(best_accs, best_full_accs)):
        args.logger.info(f"Best acc for student{s_idx} [{args.student_archs[s_idx]}] (subset): {best_acc:.4f}, (full): {best_full_acc:.4f}")

if __name__ == '__main__':
    main()