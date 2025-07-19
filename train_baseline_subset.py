import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifiar100_subset import get_cifar100_dataloaders_subset
from helper.util import save_dict_to_json, adjust_learning_rate_cifar
from helper.loops import train_vanilla as train, validate
from utils import set_logger

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
    parser.add_argument('--class_list', type=str, default=None, help='comma separated class indices, e.g. 0,1,2,3')
    # multiprocessing
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    opt = parser.parse_args()
    # 处理 class_list
    if opt.class_list is not None:
        opt.class_list = [int(x) for x in opt.class_list.split(',')]
        # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard 

    opt.model_path = os.path.join(opt.checkpoint_dir, './teachers/models')
    opt.tb_path = os.path.join(opt.checkpoint_dir, './teachers/tensorboard')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # set the model name
    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)
    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
        
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def main():
    opt = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    # 日志和保存路径
    if opt.class_list is not None:
        model_name = f"{opt.model}_subset_{'-'.join(map(str, opt.class_list))}_trial_{opt.trial}"
    else:
        model_name = f"{opt.model}_full_trial_{opt.trial}"
    save_folder = os.path.join(opt.checkpoint_dir, model_name)
    os.makedirs(save_folder, exist_ok=True)
    log_txt = os.path.join(save_folder, 'log.txt')
    loggerx = set_logger(log_txt)
    loggerx.info(f"Args: {opt}")

    # 数据加载
    if opt.class_list is not None:
        train_loader, test_loader_subset, _, _ = get_cifar100_dataloaders_subset(
            class_list=opt.class_list,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            root=opt.data_folder
        )
        _, test_loader_full = get_cifar100_dataloaders(
            opt.data_folder, batch_size=opt.batch_size, num_workers=opt.num_workers
        )
        use_subset = True
    else:
        train_loader, test_loader_full = get_cifar100_dataloaders(
            opt.data_folder, batch_size=opt.batch_size, num_workers=opt.num_workers
        )
        test_loader_subset = None
        use_subset = False

    # 模型
    model = model_dict[opt.model](num_classes=100)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    cudnn.benchmark = True

    best_acc_subset = 0
    best_acc_full = 0

    for epoch in range(1, opt.epochs + 1):
        
        adjust_learning_rate_cifar(optimizer, epoch, opt)
        train_acc, train_acc_top5, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        loggerx.info(f"Epoch {epoch} train_acc {train_acc:.3f} train_loss {train_loss:.4f}")

        if use_subset:
            # 在子集测试集上评估
            test_acc_subset, test_acc_top5_subset, test_loss_subset = validate(test_loader_subset, model, criterion, opt)
            loggerx.info(f"Epoch {epoch} [Subset] test_acc {test_acc_subset:.3f} test_loss {test_loss_subset:.4f}")
            # 在全体测试集上评估
            test_acc_full, test_acc_top5_full, test_loss_full = validate(test_loader_full, model, criterion, opt)
            loggerx.info(f"Epoch {epoch} [Full] test_acc {test_acc_full:.3f} test_loss {test_loss_full:.4f}")
            # 保存子集最优模型
            if test_acc_subset > best_acc_subset:
                best_acc_subset = test_acc_subset
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc_subset': best_acc_subset,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, os.path.join(save_folder, f"{opt.model}_best_subset.pth"))
                save_dict_to_json({
                    'test_loss_subset': float(test_loss_subset),
                    'test_acc_subset': float(test_acc_subset),
                    'test_loss_full': float(test_loss_full),
                    'test_acc_full': float(test_acc_full),
                    'epoch': epoch
                }, os.path.join(save_folder, "test_best_metrics_subset.json"))
                loggerx.info("Saved best model (by subset acc)")
            # 保存全体最优模型
            if test_acc_full > best_acc_full:
                best_acc_full = test_acc_full
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc_full': best_acc_full,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, os.path.join(save_folder, f"{opt.model}_best_full.pth"))
                save_dict_to_json({
                    'test_loss_subset': float(test_loss_subset),
                    'test_acc_subset': float(test_acc_subset),
                    'test_loss_full': float(test_loss_full),
                    'test_acc_full': float(test_acc_full),
                    'epoch': epoch
                }, os.path.join(save_folder, "test_best_metrics_full.json"))
                loggerx.info("Saved best model (by full acc)")
        else:
            # 只在全体测试集上评估
            test_acc_full, test_acc_top5_full, test_loss_full = validate(test_loader_full, model, criterion, opt)
            loggerx.info(f"Epoch {epoch} [Full] test_acc {test_acc_full:.3f} test_loss {test_loss_full:.4f}")
            # 保存全体最优模型
            if test_acc_full > best_acc_full:
                best_acc_full = test_acc_full
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc_full': best_acc_full,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, os.path.join(save_folder, f"{opt.model}_best_full.pth"))
                save_dict_to_json({
                    'test_loss_full': float(test_loss_full),
                    'test_acc_full': float(test_acc_full),
                    'epoch': epoch
                }, os.path.join(save_folder, "test_best_metrics_full.json"))
                loggerx.info("Saved best model (by full acc)")

    loggerx.info(f"Best subset acc: {best_acc_subset:.4f}, Best full acc: {best_acc_full:.4f}")

if __name__ == '__main__':
    main() 
