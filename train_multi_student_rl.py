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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from loguru import logger


from train_loops import test
from train_loops_multi import train_multi_students, test_multi_students
from torch.utils.tensorboard import SummaryWriter
from models import model_dict
from setting import teacher_model_path_dict
from dataset.cifar100 import get_cifar100_dataloaders
from utils import set_logger
from models.util import Regress, TransFeat


parser = argparse.ArgumentParser(description='PyTorch Multi-Student Training')
parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_imagenet')
parser.add_argument('--student-archs', default=['resnet18_imagenet'], type=str, nargs='+', 
                    help='list of student architectures to train simultaneously')
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--dynamic', action='store_true', help="use dynamic weight aggregation strategy")
parser.add_argument('--ce-weight', type=float, default=1, help='ce loss coefficient')
parser.add_argument('--kd-weight', type=float, default=1, help='kd loss coefficient')
parser.add_argument('--feat-weight', type=float, default=5, help='feature loss coefficient')
# parser.add_argument('--inter-student-weight', type=float, default=0.5, help='inter-student distillation weight')

import models
from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num
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


def get_tensorboard_path(path):
    time_stamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    write_path = path + '/' + time_stamp
    os.makedirs(write_path)
    return write_path


def main():
    args = parser.parse_args()
    args.teacher_name_str = "_".join(args.teacher_name_list)
    args.student_arch_str = "_".join(args.student_archs)
    logger.info('args.teacher_name_str: {}', args.teacher_name_str)
    logger.info('args.student_arch_str: {}', args.student_arch_str)
    args.teacher_num = len(args.teacher_name_list)
    args.student_num = len(args.student_archs)

    args.model_name = args.student_arch_str + '_'+ args.dataset+ '_'+ 'multi_student_rl'+'_'+ args.trial+'_'+str(args.teacher_num)+'_'+args.teacher_name_str

    info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    info = args.model_name + info_time
    logger.info('===> info is : {}', info)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    logger.info('===>args.checkpoint_dir is : {}', args.checkpoint_dir)
    if args.rank == 0 :
        args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')
        args.logger = set_logger(args.log_txt)
        args.logger.info("==========\nArgs:{}\n==========".format(args))

    # 配置loguru
    logger.remove()  # 移除默认的处理器
    logger.add(args.log_txt, rotation="500 MB", retention="10 days", level="INFO")
    logger.add(lambda msg: print(msg), level="INFO")  # 同时输出到控制台

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        logger.warning('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.gpu, 1, args)


def get_multi_agent(teacher_models, student_models, args):
    """为多个学生模型获取智能体"""
    teacher_num = len(teacher_models)
    student_num = len(student_models)
    
    x = torch.rand(args.res).cuda()
    
    # 获取教师模型的特征维度信息
    teacher_feature_dims = []
    teacher_policy_input_size = []
    
    for t in teacher_models:
        feature, logits = t(x, is_feat=True)
        teacher_feature_dims.append(feature[-2].size())
        teacher_policy_input_size.append(feature[-1].size(1) + logits.size(1) + 3)
    
    # 为每个学生模型创建智能体
    agents = []
    for s_idx in range(student_num):
        agent = model_dict['PolicyTrans'](teacher_policy_input_size, teacher_num, args.dynamic).cuda()
        agents.append(agent)
    
    return agents, teacher_feature_dims


def get_multi_feat_trans(student_models, teacher_feature_dims, args):
    """为多个学生模型获取特征转换器"""
    feat_trans_list = []
    
    for s_idx, model in enumerate(student_models):
        model.eval()
        with torch.no_grad():
            s_feat, s_logits = model(torch.rand(args.res).cuda(), is_feat=True)
        s_feat_dim = s_feat[-2].size()
        feat_trans = TransFeat(s_feat_dim, teacher_feature_dims).cuda()
        feat_trans_list.append(feat_trans)
        
    return feat_trans_list


def load_teacher(model_path, n_cls, model_t, gpu=None):
    """加载教师模型"""
    model = model_dict[model_t](num_classes=n_cls).cuda()
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % gpu}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    model.eval()
    for t_n, t_p in model.named_parameters():
        t_p.requires_grad = False
    return model


def load_teacher_list(args):
    """加载教师模型列表"""
    logger.info('==> loading teacher model list')
    teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], args.n_cls, model_name, args.gpu)
                         for model_name in args.teacher_name_list]
    logger.info('==> done')
    return teacher_model_list

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    # 设置数据集参数
    if args.dataset.startswith('cifar100'):
        args.n_cls = 100
        args.res = (1, 3, 32, 32)
    elif args.dataset.startswith('imagenet'):
        args.n_cls = 1000
        args.res = (1, 3, 224, 224)
    
    # 加载教师模型
    teacher_models = load_teacher_list(args)
    
    # 加载多个学生模型
    student_models = []
    for arch in args.student_archs:
        model = model_dict[arch](num_classes=args.n_cls).cuda()
        student_models.append(model)
    
    print(f'======> loaded {len(student_models)} student models')
    
    # 获取智能体和特征转换器
    agents, teacher_feature_dims = get_multi_agent(teacher_models, student_models, args)
    args.t_feat_dims = teacher_feature_dims
    print("===> get agents finish......")
    
    feat_trans_list = get_multi_feat_trans(student_models, teacher_feature_dims, args)
    print("===> get feat_trans_list finish......")
    
    # 设置设备
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # 损失函数
    criterion_list = nn.ModuleList([])
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_div = DistillKL(args.kd_T).to(device)
    criterion_list.append(criterion_ce)
    criterion_list.append(criterion_div)
    
    # 优化器 - 包含所有学生模型和特征转换器
    trainable_list = nn.ModuleList([])
    for model in student_models:
        trainable_list.append(model)
    for feat_trans in feat_trans_list:
        trainable_list.append(feat_trans)
    
    optimizer = optim.SGD(trainable_list.parameters(),
                         lr=args.init_lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    
    # 智能体优化器
    agent_params = []
    for agent in agents:
        agent_params.extend(list(agent.parameters()))
    
    agent_optimizer = optim.SGD(agent_params, lr=0.1)
    
    # 加载数据
    train_loader, val_loader = get_cifar100_dataloaders(data_folder=args.data,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.workers)
    
    # 训练前测试教师模型性能
    t_results = []
    for t_model in teacher_models:
        acc = test(0, t_model, device, val_loader, criterion_ce, args, verbose=False)
        t_results.append(round(acc, 2))
    args.logger.info('Teacher accuracy: ' + str(t_results))
    
    # 训练循环
    best_accs = [0.] * len(student_models)
    
    for epoch in range(args.start_epoch, args.epochs):
        # 训练多个学生模型
        train_multi_students(train_loader, student_models, criterion_list, optimizer, epoch, device, args, 
                           agents, feat_trans_list, teacher_models, agent_optimizer)
        
        # 测试多个学生模型
        accs = test_multi_students(epoch, student_models, device, val_loader, criterion_ce, args)
        
        # 保存模型
        for s_idx, (model, acc) in enumerate(zip(student_models, accs)):
            state = {
                'epoch': epoch + 1,
                'arch': args.student_archs[s_idx],
                'model': model.state_dict(),
                'acc': acc,
                'optimizer': optimizer.state_dict()
            }
            
            model_name = f"{args.student_archs[s_idx]}_student_{s_idx}"
            torch.save(state, os.path.join(args.checkpoint_dir, f'{model_name}.pth.tar'))
            
            is_best = False
            if best_accs[s_idx] < acc:
                best_accs[s_idx] = acc
                is_best = True
            
            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, f'{model_name}.pth.tar'),
                              os.path.join(args.checkpoint_dir, f'{model_name}_best.pth.tar'))
    
    # 最终评估所有学生模型
    args.logger.info('Final evaluation of all student models:')
    for s_idx, (model, best_acc) in enumerate(zip(student_models, best_accs)):
        model_name = f"{args.student_archs[s_idx]}_student_{s_idx}"
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, f'{model_name}_best.pth.tar'))
        model.load_state_dict(checkpoint['model'])
        final_acc = test(epoch, model, device, val_loader, criterion_ce, args, verbose=False)
        args.logger.info(f'Student {s_idx} ({args.student_archs[s_idx]}) - Final Test Accuracy: {final_acc:.2f}%')



if __name__ == '__main__':
    main()
