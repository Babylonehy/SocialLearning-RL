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
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


from train_loops import train, test, train_cycle
from torch.utils.tensorboard import SummaryWriter
from models import model_dict
from setting import  teacher_model_path_dict
from dataset.cifar100 import get_cifar100_dataloaders
from utils import set_logger
from models.util import Regress, TransFeat
from helper.loops import train_vanilla

from loguru import logger
import sys
import wandb

# Set up logging
# logger.remove()
# logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_imagenet')
##TODO: add from students list
parser.add_argument('--students-list', metavar='STUDENTS', nargs='+', default=['resnet18_imagenet'],)

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
                         'using Data Parallel or Distributed Data Parallel')  # 32*2
                         
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
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--dynamic', action='store_true', help="use dynamic weight aggregation strategy")
parser.add_argument('--ce-weight', type=float, default=1, help='ce loss coefficient')
parser.add_argument('--kd-weight', type=float, default=1, help='kd loss coefficient')
parser.add_argument('--feat-weight', type=float, default=5, help='kd loss coefficient')


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

parser.add_argument("--debugpy", action="store_true", help="Enable debugpy for remote debugging")
parser.add_argument('--overlap', type=float, default=0.2, help='overlap ratio for subset training')
parser.add_argument('--warmup-ratio', type=float, default=0.1, help='warmup ratio for training')

def get_tensorboard_path(path):
    time_stamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    write_path = path + '/' + time_stamp
    os.makedirs(write_path)
    return write_path

def main():
    args = parser.parse_args()

    
    if args.debugpy:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")
        # Wait for the debugger to attach before proceeding
        debugpy.wait_for_client()
    
    # args.teacher_name_str = "_".join(args.teacher_name_list)
    # print('args.teacher_name_str', args.teacher_name_str)
    # args.teacher_num = len(args.teacher_name_list)
    
    args.model_name_list = [s+"_"+args.dataset+ '_'+ 'rl'+'_'+ args.trial for s in args.students_list]
    
    # args.model_name = args.arch + '_'+ args.dataset+ '_'+ 'rl'+'_'+ args.trial+'_'+str(args.teacher_num)+'_'+args.teacher_name_str

    info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    info = "-".join([s for s in args.students_list]) + info_time
    print(f'===> info is : {info}')
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    print(f'===>args.checkpoint_dir is : {args.checkpoint_dir}')

    # if args.rank == 0 :
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
    
    if args.gpu is not None :
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed # True
    print(f'======> args.distributed is {args.distributed}')

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    print(f'======> ngpus_per_node is {ngpus_per_node}') 
    print("waitting for wandb init...")
    wandb.init(
        project="SL",  # 替换为你的项目名
        name="-".join([s for s in args.students_list]) + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        config=vars(args)
    )
    print("wandb init finished.")
    args.logger.info(f'===> wandb init finished, project: SL, name: {wandb.run.name}')
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)  


def get_agent(teacher_models, args):
    teacher_num = len(teacher_models)
    x = torch.rand(args.res).cuda()
    logits_dim = 0
    feature_dim = 0
    feature_dims = []
    policy_input_size = []
    for t in teacher_models :
        feature, logits = t(x, is_feat=True)
        logits_dim += logits.size(1)
        feature_dim += feature[-1].size(1)
        feature_dims.append(feature[-2].size())
        policy_input_size.append(feature[-1].size(1) + logits.size(1) + 3)
        
    agent = model_dict['PolicyTrans'](policy_input_size, teacher_num, args.dynamic).cuda()
    return agent, feature_dims

def get_agents(train_model_dict, args):
    model_num = len(train_model_dict.keys())
    x = torch.rand(args.res).cuda()
    agents_dict = { }
    for k,student_model in train_model_dict.items() :
        logits_dim = 0
        feature_dim = 0
        feature_dims = []
        policy_input_size = []
        for kk,teach_model in train_model_dict.items() :
            if k == kk :
                continue
            else :
                feature, logits = teach_model(x, is_feat=True)
                logits_dim += logits.size(1)
                feature_dim += feature[-1].size(1)
                feature_dims.append(feature[-2].size())
                policy_input_size.append(feature[-1].size(1) + logits.size(1) + 3)
                
        agents_dict[k] = (model_dict['PolicyTrans'](policy_input_size, model_num-1, args.dynamic).cuda(), feature_dims)
    
    return agents_dict

def get_feat_trans(model, args):
    model.eval()
    with torch.no_grad():
        s_feat, s_logits = model(torch.rand(args.res).cuda(), is_feat=True)
    args.s_feat_dim = s_feat[-2].size()
    return TransFeat(args.s_feat_dim, args.t_feat_dims).cuda()

def get_all_feat_trans(train_model_dict, args):
    """
    为 train_model_dict 里的每个 model 构建一个 TransFeat。
    每次以当前 model 为 student，其余为 teacher。
    返回 dict: {student_name: TransFeat}
    """
    feat_trans_dict = {}
    model_names = list(train_model_dict.keys())
    for student_name in model_names:
        student_model = train_model_dict[student_name]
        # 以当前 model 为 student
        student_model.eval()
        with torch.no_grad():
            s_feat, s_logits = student_model(torch.rand(args.res).cuda(), is_feat=True)
        s_feat_dim = s_feat[-2].size()
        # 其余 model 为 teacher
        teacher_feat_dims = []
        for teacher_name in model_names:
            if teacher_name == student_name:
                continue
            teacher_model = train_model_dict[teacher_name]
            teacher_model.eval()
            with torch.no_grad():
                t_feat, t_logits = teacher_model(torch.rand(args.res).cuda(), is_feat=True)
            teacher_feat_dims.append(t_feat[-2].size())
        # 构建 TransFeat
        feat_trans_dict[student_name] = TransFeat(s_feat_dim, teacher_feat_dims).cuda()
    return feat_trans_dict

def fix_random(seed=42, cudnn_deterministic=True):
    """
    Fix random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = cudnn_deterministic
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
def main_worker(gpu, ngpus_per_node, args):
    # fix random seed and cudnn
    fix_random(seed=42)
    
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
        
        
    # def load_teacher(model_path, n_cls, model_t, opt=None):
    #     model = model_dict[model_t](num_classes=n_cls).cuda()
    #     map_location = None if opt.gpu is None else {'cuda:0': 'cuda:%d' % (opt.gpu if opt.multiprocessing_distributed else 0)}
    #     model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    #     model.eval()
    #     for t_n, t_p in model.named_parameters():
    #         t_p.requires_grad = False
    #     return model


    # def load_teacher_list(opt):
    #     print('==> loading teacher model list')
    #     teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], args.n_cls, model_name, opt)
    #                         for model_name in opt.teacher_name_list]
    #     print('==> done')
    #     return teacher_model_list

    if args.dataset.startswith('cifar100'):
        args.n_cls = 100
        args.res = (1, 3, 32, 32)
    elif args.dataset.startswith('imagenet'):
        args.n_cls = 1000
        args.res = (1, 3, 224, 224)
    
    # teacher_models = load_teacher_list(args)
    
    ##### load student model #####
    model_train_dicts = {}
    for student_name in args.students_list:
        print(f'======> load student model {student_name} start')
        model_train_dicts[student_name] = model_dict[student_name](num_classes=args.n_cls).cuda()
    
    # model = model_dict[args.arch](num_classes=args.n_cls).cuda()
    
    args.start_epoch = 0
    
    # if len(args.resume) != 0:
    #     map_location = None if args.gpu is None else 'cuda:%d' % (args.gpu if args.multiprocessing_distributed else 0)
    #     model_info_dict = torch.load(args.resume, map_location=map_location)
    #     model.load_state_dict(model_info_dict['model'])
    #     args.start_epoch = model_info_dict['epoch']
        
    print('======> load student model finish')
    agents_dict = get_agents(model_train_dicts, args)
    print("===> get agent finish......")
    
    feat_trans_dict = get_all_feat_trans(model_train_dicts, args)
    print("===> get feat_trans finish......")
        
    # if args.distributed:
    #     if torch.cuda.is_available():
    #         if args.gpu is not None:
    #             torch.cuda.set_device(args.gpu)
    #             for teacher in teacher_models :
    #                 teacher.cuda(args.gpu)
    #             agent.cuda(args.gpu)
    #             model.cuda(args.gpu)
    #             args.batch_size = int(args.batch_size / ngpus_per_node)
    #             args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #             model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu]) 
    #             # feat_trans = torch.nn.parallel.DistributedDataParallel(feat_trans, device_ids=[args.gpu])
    #             for teacher in teacher_models : 
    #                 teacher = torch.nn.parallel.DistributedDataParallel(teacher,device_ids=[args.gpu])
    #             agent = torch.nn.parallel.DistributedDataParallel(agent,device_ids=[args.gpu])
                
    #         else:
    #             model.cuda()
    #             model = torch.nn.parallel.DistributedDataParallel(model)
    #             for teacher in teacher_models :
    #                 teacher = teacher.cuda()
    #             agent = agent.cuda()
    #             agent = torch.nn.parallel.DistributedDataParallel(agent)  
                
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        for model in model_train_dicts:
            model_train_dicts[model] = model_train_dicts[model].cuda(args.gpu)
            agents_dict[model] = agents_dict[model][0].cuda(args.gpu)
            feat_trans_dict[model] = feat_trans_dict[model].cuda(args.gpu)
            
        # model = model.cuda(args.gpu)
        # agent = agent.cuda(args.gpu)
        # for teacher in teacher_models :
        #     teacher = teacher.cuda(args.gpu)
    else:
        # model = torch.nn.DataParallel(model).cuda()
        for model in model_train_dicts:
            model_train_dicts[model] = model_train_dicts[model].cuda(args.gpu)
            agents_dict[model] = agents_dict[model][0].cuda(args.gpu)
            feat_trans_dict[model] = feat_trans_dict[model].cuda(args.gpu)
    
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    ################### loss function and optimizer ###################
   
          
    criterion_list = nn.ModuleList([])
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_div = DistillKL(args.kd_T).to(device)
    criterion_list.append(criterion_ce)
    criterion_list.append(criterion_div)

    # trainable_list = nn.ModuleList([])
    # trainable_list.append(model)
    # trainable_list.append(feat_trans)
    model_optimizer_dict = {}
    agent_optimizer_dict = {}
    for k, v in model_train_dicts.items():
        trainable_list = nn.ModuleList([])
        trainable_list.append(v)
        trainable_list.append(feat_trans_dict[k])
        model_optimizer_dict[k] = optim.SGD(trainable_list.parameters(),
                                    lr=args.init_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
        
        agent_optimizer_dict[k] = optim.SGD(agents_dict[k].parameters(),
                                    lr=args.init_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
        
            
    # optimizer = optim.SGD(trainable_list.parameters(),
                        # lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    
    
    # agent_optimizer = optim.SGD(agent.parameters(), lr=0.1, )

    ################### load data ###################
    # train_loader, val_loader = get_cifar100_dataloaders(data_folder=args.data,
    #                                                     batch_size=args.batch_size,
    #                                                     num_workers=args.workers)
    # dataloader
    model_dataloader_dict = {}
    if args.dataset == 'cifar100':
        if hasattr(args, 'overlap') and args.overlap < 1:
            # split cifar100 dataset into subsets
            from dataset.cifiar100_subset import create_cifar100_subset_dataloaders,get_cifar100_subset_class
            subset = get_cifar100_subset_class(overlap=args.overlap, model_num=len(model_train_dicts.keys()), num_classes=args.n_cls)
            for idx, k in enumerate(model_train_dicts.keys()):
                print(f'{k}===> subset_class is {subset[idx]}')
                train_loader, val_loader = create_cifar100_subset_dataloaders(subset[idx], batch_size=args.batch_size, num_workers=args.workers, root=args.data)
                model_dataloader_dict[k] = (train_loader, val_loader)
            # Also get full test loader for evaluation
            _, full_val_loader = get_cifar100_dataloaders(args.data, batch_size=args.batch_size, num_workers=args.workers)
            full_test_loader = full_val_loader
        elif hasattr(args, 'overlap') and args.overlap == 1:
            train_loader, val_loader = get_cifar100_dataloaders(args.data, batch_size=args.batch_size, num_workers=args.workers)
            full_test_loader = val_loader
            for k in model_train_dicts.keys():
                model_dataloader_dict[k] = (train_loader, val_loader)
        else:
            raise ValueError("Invalid dataset or overlap value. Please check your arguments.")
        
    ################### train model ###################
    ## Warmup phase 
    best_acc = {}
    warmup_epochs = int(args.epochs * args.warmup_ratio) if hasattr(args, 'warmup_ratio') else 0
    args.logger.info(f'===> Total epochs: {args.epochs}')
    args.logger.info(f'====Start warmup training for {warmup_epochs} epochs====')
    for epoch in range(warmup_epochs):
        for k, (train_loader, val_loader) in model_dataloader_dict.items():
            print(f'===> Warmup Epoch {epoch} Model {k} Start')
            train_vanilla(epoch, train_loader, model_train_dicts[k], criterion_ce, model_optimizer_dict[k], args)
            acc = test(epoch, model_train_dicts[k], device, val_loader, criterion_ce, args)
            acc_full  = test(epoch, model_train_dicts[k], device, full_test_loader, criterion_ce, args)
            args.logger.info(f'#######Warmup Epoch {epoch} Model {k} Acc: {acc:.2f}, Full Acc: {acc_full:.2f}')
            wandb.log({
                f'{k}/warmup_acc': acc,
                f'{k}/warmup_full_acc': acc_full,
                'epoch': epoch
            })
            if best_acc.get(k, 0) < acc:
                best_acc[k] = acc
                # Save the best model
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'model': model_train_dicts[k].module.state_dict() if args.distributed else model_train_dicts[k].state_dict(),
                    'acc': acc,
                    'optimizer': model_optimizer_dict[k].state_dict()
                }, os.path.join(args.checkpoint_dir, f'{k}_best_warmup.pth.tar'))
    args.logger.info(f'====Warmup training finished====')
    
    args.logger.info(f'===> Best accuracy after warmup: {best_acc}')
    # best_acc = 0.  # best test accuracy
    
    # t_results = []
    # for t_model in model_dataloader_dict.keys():
    #     acc = test(0, t_model, device, val_loader, criterion_ce, args, verbose=False)
    #     t_results.append(round(acc, 2))
    # args.logger.info('Teacher accruacy: '+ str(t_results))
    best_acc = {}
    best_acc_full = {}
    
    for epoch in range(warmup_epochs, args.epochs) :
        
        train_cycle(
            model_dataloader_dict,  # dict: {model_name: (train_loader, val_loader)}
            model_train_dicts,      # dict: {model_name: model}
            model_optimizer_dict,   # dict: {model_name: optimizer}
            agent_optimizer_dict,   # dict: {model_name: agent_optimizer}
            feat_trans_dict,        # dict: {model_name: feat_trans}
            agents_dict,             # dict: {model_name: agent}
            criterion_list,         # list: [ce, div, ...]
            epoch, device, args,
            full_test_loader,  # full validation loader for evaluation
            best_acc,
            best_acc_full,
            wandb=wandb  # 传递wandb
        )
        
        
        # train(train_loader, model, criterion_list, optimizer, epoch, device, args, agent, feat_trans, teacher_models, agent_optimizer)
        
        # acc = test(epoch, model, device, val_loader, criterion_ce, args)

        # if args.rank == 0 :
        #     state = {
        #             'epoch' : epoch + 1,
        #             'arch' : args.arch, 
        #             'model': model.module.state_dict() if args.distributed else model.state_dict() ,
        #             'acc': acc,
        #             'epoch': epoch,
        #             'optimizer': optimizer.state_dict()
        #     }

        #     torch.save(state, os.path.join(args.checkpoint_dir, args.arch+'.pth.tar'))

        #     is_best = False
        #     if best_acc < acc:
        #         best_acc = acc
        #         is_best = True
        #     if is_best:
        #         shutil.copyfile(os.path.join(args.checkpoint_dir, args.arch + '.pth.tar'),
        #                             os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'))

    
    if args.rank == 0 :
        args.logger.info('Evaluate the best model:')
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'),
                                    map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        top1_acc = test(epoch, model, device, val_loader, criterion_ce, args)
        args.logger.info('Test top-1 best_accuracy: {}'.format(top1_acc))
        args.logger.info('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir,  args.arch + '_best.pth.tar')))
    wandb.finish()

if __name__ == '__main__' :
     main()





    


