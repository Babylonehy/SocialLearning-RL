from __future__ import print_function

import os
import argparse
import random
import time
import warnings

import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate_cifar
from helper.loops import train_vanilla as train, validate
from utils import set_logger
import torch

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # baisc
    parser.add_argument('--print-freq', type=int, default=200, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('--data-folder', type=str, default='/data/winycg/dataset', help='dataset path')
    parser.add_argument('--checkpoint-dir', type=str, default='/data/winycg/checkpoints/mkd_checkpoints/', help='checkpoint dir')
    
    parser.add_argument('-t', '--trial', type=str, default='0', help='the experiment id')
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--class-list', type=str, default=None, help='comma separated class indices for subset training (e.g. "1,2,3")')

    # multiprocessing
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    
    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard 
    if opt.checkpoint_dir:
        opt.model_path = os.path.join(opt.checkpoint_dir, 'teachers', 'models')
        opt.tb_path = os.path.join(opt.checkpoint_dir, 'tensorboard')
    else:
        opt.model_path = os.path.join('./output', 'teachers', 'models')
        opt.tb_path = os.path.join('./runs')
    print("saving models to: ", opt.model_path)
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # set the model name
    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)
    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

        
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # tensorboard 日志目录直接放在 save_folder/tensorboard
    opt.tb_folder = os.path.join(opt.save_folder, 'tensorboard')
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    return opt

best_acc = 0
total_time = time.time()

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
def main():
    fix_random(seed=42, cudnn_deterministic=True)
    
    opt = parse_option()
    log_txt =  os.path.join(opt.save_folder, 'log.txt')
    opt.loggerx = set_logger(log_txt)
    opt.loggerx.info("==========\nArgs:{}\n==========".format(opt))

    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)
    
    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = int(gpu)
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

    # model
    n_cls = {
        'cifar100': 100,
    }.get(opt.dataset, None)
    
    model = model_dict[opt.model](num_classes=n_cls)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)  
                model = model.cuda(opt.gpu)
                criterion = criterion.cuda(opt.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                opt.batch_size = int(opt.batch_size / ngpus_per_node)
                opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion = criterion.cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model).cuda()
            else:
                model = model.cuda()


    cudnn.benchmark = True

    # dataloader
    if opt.dataset == 'cifar100':
        if hasattr(opt, 'class_list') and opt.class_list is not None:
            from dataset.cifiar100_subset import create_cifar100_subset_dataloaders
            class_list = [int(x) for x in opt.class_list.split(',')]
            train_loader, val_loader = create_cifar100_subset_dataloaders(
                class_list, batch_size=opt.batch_size, num_workers=opt.num_workers, root=opt.data_folder)
            # Also get full test loader for evaluation
            _, full_val_loader = get_cifar100_dataloaders(opt.data_folder, batch_size=opt.batch_size, num_workers=opt.num_workers)
            full_test_loader = full_val_loader
        else:
            train_loader, val_loader = get_cifar100_dataloaders(opt.data_folder, batch_size=opt.batch_size, num_workers=opt.num_workers)
            full_test_loader = val_loader
    else:
        raise NotImplementedError(opt.dataset)
    tic = time.time()
    # routine
    # tensorboard writer
    writer = SummaryWriter(opt.tb_folder)
    for epoch in range(1, opt.epochs + 1):

        if opt.dataset in ['cifar100']:
            adjust_learning_rate_cifar(optimizer, epoch, opt)
        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            opt.loggerx.info(' * Epoch {}, Acc@1 {:.3f}, Acc@5 {:.3f}, train_loss {:.4f}, Time {:.2f}'.format(epoch, train_acc, train_acc_top5, train_loss, time2 - time1))

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        # 新增：在全量测试集上评估
        if hasattr(opt, 'class_list') and opt.class_list is not None:
            full_test_acc, full_test_acc_top5, full_test_loss = validate(full_test_loader, model, criterion, opt)
        else:
            full_test_acc, full_test_acc_top5, full_test_loss = test_acc, test_acc_top5, test_loss

        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()
            full_val_loader.reset()
            
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            opt.loggerx.info(' ** Acc@1 {:.3f}, Acc@5 {:.3f}, test_loss {:.4f}'.format(test_acc, test_acc_top5, test_loss))
            if hasattr(opt, 'class_list') and opt.class_list is not None:
                opt.loggerx.info(' ** [Full Test] Acc@1 {:.3f}, Acc@5 {:.3f}, test_loss {:.4f}'.format(full_test_acc, full_test_acc_top5, full_test_loss))

            # save the best model (subset)
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model.module.state_dict() if opt.multiprocessing_distributed else model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, '{}_best_subset.pth'.format(opt.model))
                test_metrics = { 'test_loss': float('%.2f' % test_loss),
                                 'test_acc': float('%.2f' % test_acc),
                                 'test_acc_top5': float('%.2f' % test_acc_top5),
                                 'epoch': epoch,
                                 'full_test_loss': float('%.2f' % full_test_loss),
                                 'full_test_acc': float('%.2f' % full_test_acc),
                                 'full_test_acc_top5': float('%.2f' % full_test_acc_top5),
                               }
                save_dict_to_json(test_metrics, os.path.join(opt.save_folder, "test_best_subset_metrics.json"))
                print('saving the best subset model!')
                torch.save(state, save_file)

            # save the best model (full test)
            if hasattr(opt, 'class_list') and opt.class_list is not None:
                if not hasattr(opt, 'best_full_acc'):
                    opt.best_full_acc = 0
                if full_test_acc > opt.best_full_acc:
                    opt.best_full_acc = full_test_acc
                    state_full = {
                        'epoch': epoch,
                        'model': model.module.state_dict() if opt.multiprocessing_distributed else model.state_dict(),
                        'best_full_acc': opt.best_full_acc,
                        'optimizer': optimizer.state_dict(),
                    }
                    save_file_full = os.path.join(opt.save_folder, '{}_best_full.pth'.format(opt.model))
                    test_metrics_full = { 'test_loss': float('%.2f' % test_loss),
                                         'test_acc': float('%.2f' % test_acc),
                                         'test_acc_top5': float('%.2f' % test_acc_top5),
                                         'epoch': epoch,
                                         'full_test_loss': float('%.2f' % full_test_loss),
                                         'full_test_acc': float('%.2f' % full_test_acc),
                                         'full_test_acc_top5': float('%.2f' % full_test_acc_top5),
                                       }
                    save_dict_to_json(test_metrics_full, os.path.join(opt.save_folder, "test_best_full_metrics.json"))
                    print('saving the best full test model!')
                    torch.save(state_full, save_file_full)
            print('Epoch Total time(hms):', time.strftime("%H:%M:%S", time.gmtime(time.time() - total_time)))
            # 新增：打印预计训练结束时间
            elapsed = time.time() - total_time
            avg_epoch_time = elapsed / epoch
            remaining_epochs = opt.epochs - epoch
            est_remaining = avg_epoch_time * remaining_epochs
            est_end_time = time.time() + est_remaining
            print('Estimated finish time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(est_end_time)))
        
        # tensorboard: 记录训练loss和acc
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc1', train_acc, epoch)
        writer.add_scalar('train/acc5', train_acc_top5, epoch)
        writer.add_scalar('val/loss', test_loss, epoch)
        writer.add_scalar('val/acc1', test_acc, epoch)
        writer.add_scalar('val/acc5', test_acc_top5, epoch)
        if hasattr(opt, 'class_list') and opt.class_list is not None:
            writer.add_scalar('full_val/loss', full_test_loss, epoch)
            writer.add_scalar('full_val/acc1', full_test_acc, epoch)
            writer.add_scalar('full_val/acc5', full_test_acc_top5, epoch)

        # 保存前几张测评图片及其预测和gt
        if epoch == 1 or epoch % 10 == 0:
            import torchvision
            import numpy as np
            import matplotlib.pyplot as plt
            from io import BytesIO
            from dataset.cifar100 import CIFAR100_CLASSES
            model.eval()
            images, labels = next(iter(val_loader))
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                outputs = model(images)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                _, preds = outputs.topk(5, 1, True, True)
            images_np = images[:8].cpu().numpy()
            images_np = np.transpose(images_np, (0, 2, 3, 1))  # NCHW->NHWC
            images_np = (images_np * 255).clip(0, 255).astype(np.uint8)
            img_list = []
            for i in range(min(8, images_np.shape[0])):
                pred_top1 = int(preds[i][0].cpu().numpy())
                gt = int(labels[i])
                pred_name = CIFAR100_CLASSES[pred_top1]
                gt_name = CIFAR100_CLASSES[gt]
                fig, ax = plt.subplots(figsize=(2.5, 3), dpi=100)
                ax.imshow(images_np[i])
                color = 'g' if pred_top1 == gt else 'r'
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(4)
                ax.axis('off')
                # 分别绘制 pred/gt
                ax.text(0.5, 1.05, f'pred: {pred_name}', color=color, fontsize=14, ha='center', va='bottom', transform=ax.transAxes)
                ax.text(0.5, 1.15, f'gt: {gt_name}', color='black', fontsize=14, ha='center', va='bottom', transform=ax.transAxes)
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2)
                plt.close(fig)
                buf.seek(0)
                img = plt.imread(buf, format='png')
                img = img[..., :3]  # 去掉alpha通道
                img = torch.from_numpy((img*255).astype(np.uint8)).permute(2,0,1).float()/255
                img_list.append(img)
            # 先收集所有图片的宽高
            shapes = [img.shape for img in img_list]
            min_h = min([s[1] for s in shapes])
            min_w = min([s[2] for s in shapes])
            # 裁剪到最小宽高
            img_list = [img[:, :min_h, :min_w] for img in img_list]
            img_grid = torchvision.utils.make_grid(img_list, nrow=4, normalize=True, scale_each=True)
            writer.add_image('val/images_with_gt_pred', img_grid, epoch)
            model.train()

    writer.close()

    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # This best accuracy is only for printing purpose.
        opt.loggerx.info('best_accuracy {:.4f}'.format(best_acc))

        # save parameters
        state = {k: v for k, v in opt._get_kwargs()}

    print('Total time(hms):', time.strftime("%H:%M:%S", time.gmtime(time.time() - total_time)))
    # 新增：打印 best subset 和 best full
    if hasattr(opt, 'class_list') and opt.class_list is not None:
        print(f'Best subset acc: {best_acc:.4f}')
        if hasattr(opt, 'best_full_acc'):
            print(f'Best full acc: {opt.best_full_acc:.4f}')
    
if __name__ == '__main__':
    main()
