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
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from train_loops import train, test
from torch.utils.tensorboard import SummaryWriter
from models import model_dict
from setting import teacher_model_path_dict
from dataset.cifar100 import get_cifar100_dataloaders
from utils import set_logger, AverageMeter, adjust_lr, DistillKL, correct_num
from models.util import Regress, TransFeat

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Distributed Training with torchrun')
    parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_imagenet')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=240, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size per process')
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
                        help='seed for initializing training')
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
    
    # 没有必要手动设置这些分布式参数，它们会由 torchrun 自动设置
    # 只保留与旧代码兼容需要的配置
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    
    return parser.parse_args()

def get_tensorboard_path(path):
    time_stamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    write_path = path + '/' + time_stamp
    os.makedirs(write_path)
    return write_path

def setup_for_distributed(is_master):
    """
    控制打印的辅助函数，仅在主进程上打印信息
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode():
    """
    初始化分布式训练环境，兼容 torchrun 和手动设置
    """
    # 优先使用 SLURM 环境变量，如果有的话
    if 'SLURM_PROCID' in os.environ:
        local_rank = int(os.environ['SLURM_LOCALID'])
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        
        # 从SLURM环境中获取主节点的地址
        hostnames = os.popen('scontrol show hostnames ' + os.environ['SLURM_JOB_NODELIST']).read().split()
        print(f"SLURM: {hostnames}")
        master_addr = hostnames[0]
        master_port = '29500'  # 可以从环境变量获取或使用固定值
        
        # 设置分布式环境变量
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
    else:
        # 使用 torchrun 自动设置的环境变量
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            print('未检测到分布式环境变量，将使用单进程运行')
            rank = 0
            world_size = 1
            local_rank = 0
    
    # 初始化分布式进程组
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",  # 对于GPU训练，使用NCCL后端
            # init_method不再需要，由环境变量MASTER_ADDR和MASTER_PORT控制
            world_size=world_size,
            rank=rank
        )
        # 设置当前设备
        torch.cuda.set_device(local_rank)
        # 控制打印
        setup_for_distributed(rank == 0)
        # 同步所有进程
        dist.barrier()
    
    return rank, world_size, local_rank

def load_teacher(model_path, n_cls, model_t):
    """加载教师模型"""
    model = model_dict[model_t](num_classes=n_cls).cuda()
    map_location = f'cuda:{torch.cuda.current_device()}'
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    model.eval()
    for t_n, t_p in model.named_parameters():
        t_p.requires_grad = False
    return model

def load_teacher_list(opt):
    """加载教师模型列表"""
    print('==> loading teacher model list')
    teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], opt.n_cls, model_name)
                       for model_name in opt.teacher_name_list]
    print('==> done')
    return teacher_model_list

def get_agent(teacher_models, args):
    """获取代理模型"""
    teacher_num = len(teacher_models)
    x = torch.rand(args.res).cuda()
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
        
    agent = model_dict['PolicyTrans'](policy_input_size, teacher_num, args.dynamic).cuda()
    return agent, feature_dims

def get_feat_trans(model, args):
    """获取特征转换模型"""
    model.eval()
    with torch.no_grad():
        s_feat, s_logits = model(torch.rand(args.res).cuda(), is_feat=True)
    args.s_feat_dim = s_feat[-2].size()
    return TransFeat(args.s_feat_dim, args.t_feat_dims).cuda()

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """保存检查点"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'model_best.pth.tar'))

def main():
    args = parse_args()
    
    # 初始化分布式环境
    rank, world_size, local_rank = init_distributed_mode()
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    args.distributed = world_size > 1
    
    # 设置模型名称和检查点目录
    args.teacher_name_str = "_".join(args.teacher_name_list)
    args.model_name = args.arch + '_'+ args.dataset+ '_'+ 'rl'+'_'+ args.trial+'_'+str(len(args.teacher_name_list))+'_'+args.teacher_name_str
    
    info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    info = args.model_name + "_" + info_time
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
    
    # 只在主进程中创建目录和设置日志
    if rank == 0:
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')
        args.logger = set_logger(args.log_txt)
        args.logger.info(f"世界大小: {world_size}, 当前进程排名: {rank}, 本地排名: {local_rank}")
        args.logger.info(f"args: {args}")
    
    # 设置随机种子，确保所有进程使用相同的初始化
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('您已选择为训练设置种子。这将开启CUDNN确定性设置，可能会显著减慢训练速度！')
    else:
        # 启用cudnn基准以提高性能
        cudnn.benchmark = True
    
    # 根据数据集设置参数
    if args.dataset.startswith('cifar100'):
        args.n_cls = 100
        args.res = (1, 3, 32, 32)
    elif args.dataset.startswith('imagenet'):
        args.n_cls = 1000
        args.res = (1, 3, 224, 224)
    
    # 加载教师模型列表
    teacher_models = load_teacher_list(args)
    
    # 创建学生模型
    model = model_dict[args.arch](num_classes=args.n_cls).cuda()
    
    # 从检查点恢复（如果有）
    if args.resume:
        if os.path.isfile(args.resume):
            if rank == 0:
                print(f"=> 从检查点加载 '{args.resume}'")
            map_location = f'cuda:{local_rank}'
            checkpoint = torch.load(args.resume, map_location=map_location)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            if rank == 0:
                print(f"=> 已加载检查点 '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            if rank == 0:
                print(f"=> 没有找到检查点 '{args.resume}'")
    
    # 获取代理模型和特征转换模型
    agent, args.t_feat_dims = get_agent(teacher_models, args)
    feat_trans = get_feat_trans(model, args)
    
    # 将模型包装为DDP模型
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        agent = torch.nn.parallel.DistributedDataParallel(agent, device_ids=[local_rank])
        feat_trans = torch.nn.parallel.DistributedDataParallel(feat_trans, device_ids=[local_rank])
    
    # 设置损失函数和优化器
    criterion_list = nn.ModuleList([])
    criterion_ce = nn.CrossEntropyLoss().cuda()
    criterion_div = DistillKL(args.kd_T).cuda()
    criterion_list.append(criterion_ce)
    criterion_list.append(criterion_div)
    
    trainable_list = nn.ModuleList([])
    trainable_list.append(model)
    trainable_list.append(feat_trans)
    
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=args.lr, momentum=args.momentum, 
                          weight_decay=args.weight_decay, nesterov=True)
    agent_optimizer = optim.SGD(agent.parameters(), lr=0.1)
    
    # 获取数据加载器，使用DistributedSampler进行分布式训练
    train_loader, val_loader = get_cifar100_dataloaders(
        data_folder=args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        distributed=args.distributed
    )
    
    # 测试教师模型的准确率
    if rank == 0:
        t_results = []
        for t_model in teacher_models:
            acc = test(0, t_model, 'cuda', val_loader, criterion_ce, args, verbose=False)
            t_results.append(round(acc, 2))
        args.logger.info('Teacher accuracy: ' + str(t_results))
    
    # 主训练循环
    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # 训练一个epoch
        train(train_loader, model, criterion_list, optimizer, epoch, 'cuda', 
              args, agent, feat_trans, teacher_models, agent_optimizer)
        
        # 测试模型
        acc = test(epoch, model, 'cuda', val_loader, criterion_ce, args)
        
        # 保存检查点（仅主进程）
        if rank == 0:
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'acc': acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }
            
            save_checkpoint(
                state, 
                is_best,
                args.checkpoint_dir,
                filename=f"{args.arch}.pth.tar"
            )
    
    # 在训练结束时评估最佳模型（仅主进程）
    if rank == 0:
        args.logger.info('评估最佳模型:')
        args.evaluate = True
        best_checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'model_best.pth.tar'),
                               map_location='cpu')
        model.load_state_dict(best_checkpoint['model'])
        top1_acc = test(args.epochs, model, 'cuda', val_loader, criterion_ce, args)
        args.logger.info(f'测试集上的最佳 top-1 准确率: {top1_acc}')
        args.logger.info(f'已从以下位置加载预训练权重: {os.path.join(args.checkpoint_dir, "model_best.pth.tar")}')
        args.logger.info(f'训练完成! 最佳精度: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
