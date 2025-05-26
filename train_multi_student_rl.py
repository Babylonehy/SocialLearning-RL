import argparse
import os
import random
import shutil
import time
import warnings
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp # Not strictly needed if using torchrun
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP


# Assuming train_loops_multi.py will contain the new train/test loops
# from train_loops_multi import train_multi_student_rl_loop, test_multi_student_rl_loop
# For now, we will define these loops within this file for simplicity,
# you can move them to a separate file later.

from models import model_dict
from setting import teacher_model_path_dict
from dataset.cifar100 import get_cifar100_dataloaders # Ensure this handles DDP
from utils import set_logger, AverageMeter, DistillKL, correct_num, adjust_lr as adjust_lr_legacy # Use your existing adjust_lr
from models.util import TransFeat

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Multi-Student RL Distillation Training')
    parser.add_argument('--data', metavar='DIR', default='./data',
                        help='path to dataset')
    parser.add_argument('--student-arch-list', metavar='ARCH', type=str, nargs='+', default=['ShuffleV2'],
                        help='student architecture(s)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=240, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                        help='mini-batch size per process')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial learning rate for students')
    parser.add_argument('--agent-lr', default=0.01, type=float, help='Initial learning rate for RL agent')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--resume-students', default=None, type=str, nargs='*',
                        help='list of paths to student checkpoints to resume from (one per student, or None)')
    parser.add_argument('--resume-agent', default=None, type=str,
                        help='path to agent checkpoint to resume from')
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    parser.add_argument('--dynamic', action='store_true', help="use dynamic weight aggregation strategy for RL agent")

    # Loss Weights
    parser.add_argument('--ce-weight', type=float, default=1.0, help='cross-entropy loss coefficient')
    parser.add_argument('--teacher-kd-weight', type=float, default=1.0, help='KD loss (logits) from teachers coefficient')
    parser.add_argument('--teacher-feat-weight', type=float, default=1.0, help='feature distillation loss from teachers coefficient') # Original was 5.0
    parser.add_argument('--mutual-kd-weight', type=float, default=0.5, help='Mutual KD loss (logits) coefficient among students')
    parser.add_argument('--mutual-feat-weight', type=float, default=0.25, help='Mutual feature loss coefficient among students (simplified)')


    # KD & Feature Distillation Params
    parser.add_argument('--kd-T', type=int, default=4, help='temperature for KD')
    parser.add_argument('--teacher-feat-kd-type', default='mse', type=str, choices=['mse', 'kl_div'], help='feature kd loss type for teacher->student')


    # RL Agent Params
    parser.add_argument('--agent-step-freq', type=int, default=100, help='RL agent optimization step frequency (in batches)')


    # Paths and Dataset
    parser.add_argument('--checkpoint-dir', default='./checkpoints_multi_student_v2', type=str, help='checkpoint directory')
    parser.add_argument('--teacher-name-list', default=['resnet32x4', 'wrn_28_4'], type=str, nargs='+', help='teacher models')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('--trial', type=str, default='ms_rl_v2_1', help='trial id for naming')


    # DDP (torchrun will handle these, but provide for awareness)
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    # local_rank will be set by torchrun
    parser.add_argument('--gpu', default=None, type=int, help='Obsolete if using torchrun. GPU id to use for single-process non-DDP.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Obsolete if using torchrun. Use multi-processing distributed training.')

    # For adjust_lr_legacy if used
    parser.add_argument('--milestones', default=[150,180,210], type=int, nargs='+', help='milestones for lr-multistep')
    parser.add_argument('--init-lr', default=0.05, type=float, help='legacy learning rate, now covered by --lr')
    parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy for adjust_lr_legacy')


    return parser.parse_args()


def setup_ddp(args):
    """Initializes DDP."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = True
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        dist.barrier() # Wait for all processes to sync up
        print(f"DDP Initialized: Rank {args.rank}/{args.world_size}, Local Rank {args.local_rank} on GPU {torch.cuda.current_device()}")
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            print(f"Single GPU mode on GPU: {args.gpu}")
        else:
            print("Single CPU mode (or single GPU if available and not specified)")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_logger(args):
    if args.rank == 0:
        student_arch_str = "_".join(args.student_arch_list)
        teacher_name_str = "_".join(args.teacher_name_list)
        model_name_for_run = f"{student_arch_str}_vs_{teacher_name_str}_{args.dataset}_mutualRL_{args.trial}"
        info_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{model_name_for_run}_{info_time}"
        
        args.current_checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
        os.makedirs(args.current_checkpoint_dir, exist_ok=True)
        
        log_file = os.path.join(args.current_checkpoint_dir, f"{run_name}_log.txt")
        logger = set_logger(log_file)
        logger.info(f"Run Name: {run_name}")
        logger.info("Effective Args: {}".format(args))
        return logger
    else:
        class DummyLogger:
            def info(self, msg): pass
            def warning(self, msg): pass
        return DummyLogger()

def load_teacher_model(model_name, n_cls, device, args):
    model_path = teacher_model_path_dict[model_name]
    teacher = model_dict[model_name](num_classes=n_cls)
    
    map_location = device
    if args.distributed and args.gpu is None: # DDP setup might mean map_location to specific device
         map_location = {'cuda:0': f'cuda:{args.local_rank}'} if torch.cuda.is_available() else 'cpu'
    elif args.gpu is not None and not args.distributed: # Single GPU non-DDP
         map_location = f'cuda:{args.gpu}'

    checkpoint = torch.load(model_path, map_location=map_location)
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    teacher.load_state_dict(new_state_dict)
    
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher

def get_model_feature_dims(model, input_res_shape, device, is_teacher=False):
    """Gets feature dimensions, specifically for features[-2] and features[-1] (embedding)."""
    dummy_input = torch.randn(input_res_shape, device=device)
    model.eval() # Ensure model is in eval mode for consistent feature extraction
    with torch.no_grad():
        features, _ = model(dummy_input, is_feat=True)
    # features[-2] is usually the last conv feature map
    # features[-1] is usually the embedding before the classifier
    return features[-2].size(), features[-1].size()


def main_worker(args): # gpu argument is implicitly args.local_rank for DDP or args.gpu for non-DDP
    logger = args.logger # Already configured in main()

    device = torch.device(f'cuda:{args.local_rank}' if args.distributed else (f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else 'cpu'))
    logger.info(f"Process {args.rank} using device: {device}")

    # Dataset
    if args.dataset == 'cifar100':
        args.n_cls = 100
        # For querying feature dims, use batch size 1 or 2 to save memory
        args.feat_query_batch_size = 2 
        args.input_res_for_query = (args.feat_query_batch_size, 3, 32, 32) # (B, C, H, W)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not configured.")

    train_loader, val_loader = get_cifar100_dataloaders(
        data_folder=args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        # distributed=args.distributed
    )

    # --- Teacher Models ---
    teacher_models = [load_teacher_model(name, args.n_cls, device, args) for name in args.teacher_name_list]
    teacher_last_conv_feat_dims = []
    teacher_embedding_dims = []
    for t_model in teacher_models:
        conv_dim, emb_dim = get_model_feature_dims(t_model, args.input_res_for_query, device, is_teacher=True)
        teacher_last_conv_feat_dims.append(conv_dim) # This is t_feat_dims for TransFeat
        teacher_embedding_dims.append(emb_dim)     # This is for agent state
    
    if args.rank == 0:
        logger.info(f"Teacher last conv feature map dims (for TransFeat): {teacher_last_conv_feat_dims}")
        logger.info(f"Teacher embedding dims (for Agent state): {teacher_embedding_dims}")

    # --- Student Models, TransFeat, Optimizers ---
    student_models_list = []
    student_feat_trans_list = []
    student_optimizers_list = []

    for i, student_arch_name in enumerate(args.student_arch_list):
        s_model = model_dict[student_arch_name](num_classes=args.n_cls).to(device)
        
        s_last_conv_feat_dim, _ = get_model_feature_dims(s_model, args.input_res_for_query, device)
        if args.rank == 0: 
            logger.info(f"Student {i} ({student_arch_name}) raw last conv feature dim: {s_last_conv_feat_dim}")

        # TransFeat for this student to align with all teachers' last_conv_feat_dims
        s_feat_trans = TransFeat(s_last_conv_feat_dim, teacher_last_conv_feat_dims).to(device)
        
        trainable_params = list(s_model.parameters()) + list(s_feat_trans.parameters())
        optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        
        if args.distributed:
            s_model = DDP(s_model, device_ids=[args.local_rank], find_unused_parameters=False)
            s_feat_trans = DDP(s_feat_trans, device_ids=[args.local_rank], find_unused_parameters=False)

        student_models_list.append(s_model)
        student_feat_trans_list.append(s_feat_trans)
        student_optimizers_list.append(optimizer)

        # Resume student model if path provided
        if args.resume_students and len(args.resume_students) > i and args.resume_students[i]:
            chkpt_path = args.resume_students[i]
            if os.path.isfile(chkpt_path):
                logger.info(f"Loading checkpoint for student {i} from {chkpt_path}")
                checkpoint = torch.load(chkpt_path, map_location=device)
                model_to_load = s_model.module if args.distributed else s_model
                model_to_load.load_state_dict(checkpoint['model'])
                if 'feat_trans' in checkpoint:
                    feat_trans_to_load = s_feat_trans.module if args.distributed else s_feat_trans
                    feat_trans_to_load.load_state_dict(checkpoint['feat_trans'])
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = max(args.start_epoch, checkpoint.get('epoch', 0)) # Take latest start epoch
            else:
                logger.warning(f"Resume path for student {i} not found: {chkpt_path}")
    
    # --- RL Agent ---
    # PolicyTrans input_size is a list: [teacher1_info_dim, teacher2_info_dim, ...]
    # teacher_info_dim = teacher_embedding_dim + teacher_logit_dim + 3 (student_teacher_metrics)
    agent_policy_input_dims = []
    for t_emb_dim, t_logit_count in zip(teacher_embedding_dims, [args.n_cls]*len(teacher_models)):
        # t_emb_dim is (B, C_emb, H_emb, W_emb), we need C_emb * H_emb * W_emb (flattened)
        # Or if it's already flattened (B, C_emb_flat), then t_emb_dim[1]
        emb_flat_dim = t_emb_dim[1]
        if len(t_emb_dim) > 2: # If not already flat
            emb_flat_dim = t_emb_dim[1] * t_emb_dim[2] * t_emb_dim[3]

        agent_policy_input_dims.append(emb_flat_dim + t_logit_count + 3)

    rl_agent = model_dict['PolicyTrans'](agent_policy_input_dims, len(teacher_models), args.dynamic).to(device)
    agent_optimizer = optim.SGD(rl_agent.parameters(), lr=args.agent_lr) # Use different LR for agent

    if args.resume_agent and os.path.isfile(args.resume_agent):
        logger.info(f"Loading agent checkpoint from {args.resume_agent}")
        agent_chkpt = torch.load(args.resume_agent, map_location=device)
        rl_agent.load_state_dict(agent_chkpt['agent'])
        agent_optimizer.load_state_dict(agent_chkpt['optimizer'])
        args.start_epoch = max(args.start_epoch, agent_chkpt.get('epoch',0))

    if args.distributed:
        rl_agent = DDP(rl_agent, device_ids=[args.local_rank], find_unused_parameters=False) # False is often fine

    # --- Loss Functions ---
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_kd_logits = DistillKL(args.kd_T).to(device) # For both teacher and peer logits KD

    if args.teacher_feat_kd_type == 'mse':
        # TransFeat output is a list of tensors, one per teacher. MSE will be calculated element-wise.
        criterion_teacher_feat = nn.MSELoss().to(device)
    elif args.teacher_feat_kd_type == 'kl_div':
        # This implies features are probability distributions or can be treated as such after softmax
        # This usually requires features to be positive. TransFeat's last layer is ReLU.
        from distiller_zoo import FeatureKLLoss # You might need to adapt this or ensure compatibility
        criterion_teacher_feat = FeatureKLLoss(args.kd_T).to(device) # Using same T for simplicity
    else:
        raise ValueError(f"Unsupported teacher feature KD type: {args.teacher_feat_kd_type}")

    # For simplified peer feature loss (MSE between raw student features if compatible)
    criterion_mutual_feat = nn.MSELoss().to(device)


    # --- Test Teacher Models (Rank 0 only) ---
    if args.rank == 0:
        logger.info("Initial Teacher Accuracies:")
        for i, t_model in enumerate(teacher_models):
            # Adapt test_multi_student_rl_loop to take a single model in a list for testing
            # Or create a simpler test function for single models.
            # For now, let's assume a simple test.
            # We need to modify test_multi_student_rl_loop to handle single model testing or create a new one.
            # Placeholder:
            # t_acc = simple_test(t_model, val_loader, criterion_ce, device, args)
            # logger.info(f"  Teacher {args.teacher_name_list[i]}: {t_acc:.2f}%")
            pass # Teacher testing needs a separate or adapted test loop

    # --- Training Loop ---
    best_avg_student_acc = 0.0
    student_best_accs = [0.0] * len(student_models_list)


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Adjust LR for all student optimizers
        # Assuming adjust_lr_legacy is compatible with your args structure
        # current_lrs_students = [adjust_lr_legacy(opt, epoch, args) for opt in student_optimizers_list]
        # A simpler schedule for now:
        if epoch in args.milestones:
            for opt in student_optimizers_list:
                for param_group in opt.param_groups:
                    param_group['lr'] *= 0.1 # Or args.lr_decay_rate
            # Agent LR can also be decayed
            for param_group in agent_optimizer.param_groups:
                 param_group['lr'] *= 0.1
            
        current_lrs_students_str = [f"{opt.param_groups[0]['lr']:.5f}" for opt in student_optimizers_list]
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Student LRs: {current_lrs_students_str}, Agent LR: {agent_optimizer.param_groups[0]['lr']:.5f}")
        
        train_multi_student_rl_loop(
            train_loader, student_models_list, teacher_models, rl_agent,
            student_feat_trans_list, student_optimizers_list, agent_optimizer,
            criterion_ce, criterion_kd_logits, criterion_teacher_feat, criterion_mutual_feat,
            epoch, device, args
        )

        # Validation for all students
        current_student_accs = test_multi_student_rl_loop(
            val_loader, student_models_list, criterion_ce, epoch, device, args
        )
        
        avg_acc_this_epoch = sum(current_student_accs) / len(current_student_accs) if current_student_accs else 0
        logger.info(f"Epoch {epoch+1} Val Avg Acc: {avg_acc_this_epoch:.2f}%. Individual: {[f'{a:.2f}' for a in current_student_accs]}")

        if args.rank == 0: # Save checkpoints only on rank 0
            is_overall_best = avg_acc_this_epoch > best_avg_student_acc
            if is_overall_best:
                best_avg_student_acc = avg_acc_this_epoch
                logger.info(f"*** New Best Average Student Accuracy: {best_avg_student_acc:.2f}% ***")

            # Save student models
            for s_idx, s_model_ddp in enumerate(student_models_list):
                s_model = s_model_ddp.module if args.distributed else s_model_ddp
                s_feat_trans_ddp = student_feat_trans_list[s_idx]
                s_feat_trans = s_feat_trans_ddp.module if args.distributed else s_feat_trans_ddp
                
                student_checkpoint_path = os.path.join(args.current_checkpoint_dir, f"student{s_idx}_{args.student_arch_list[s_idx]}.pth.tar")
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.student_arch_list[s_idx],
                    'model': s_model.state_dict(),
                    'feat_trans': s_feat_trans.state_dict(),
                    'optimizer': student_optimizers_list[s_idx].state_dict(),
                    'accuracy': current_student_accs[s_idx]
                }, student_checkpoint_path)

                if current_student_accs[s_idx] > student_best_accs[s_idx]:
                    student_best_accs[s_idx] = current_student_accs[s_idx]
                    shutil.copyfile(student_checkpoint_path, os.path.join(args.current_checkpoint_dir, f"student{s_idx}_{args.student_arch_list[s_idx]}_best.pth.tar"))
                if is_overall_best: # Also save if it's part of overall best epoch
                     shutil.copyfile(student_checkpoint_path, os.path.join(args.current_checkpoint_dir, f"student{s_idx}_{args.student_arch_list[s_idx]}_overall_best_epoch.pth.tar"))


            # Save agent
            agent_to_save = rl_agent.module if args.distributed else rl_agent
            agent_checkpoint_path = os.path.join(args.current_checkpoint_dir, "agent.pth.tar")
            torch.save({
                'epoch': epoch + 1,
                'agent': agent_to_save.state_dict(),
                'optimizer': agent_optimizer.state_dict(),
            }, agent_checkpoint_path)
            if is_overall_best:
                shutil.copyfile(agent_checkpoint_path, os.path.join(args.current_checkpoint_dir, "agent_best_epoch.pth.tar"))
    
    logger.info(f"Training finished. Best average student accuracy over epochs: {best_avg_student_acc:.2f}%")
    logger.info(f"Individual best student accuracies: {[f'{acc:.2f}' for acc in student_best_accs]}")


# ==========================================================================================
# Training and Testing Loops (Can be moved to train_loops_multi.py)
# ==========================================================================================

def prepare_agent_state_for_student(
    s_idx, # Current student index
    student_last_conv_feat, # Raw last conv feature of current student [B, Cs, Hs, Ws]
    student_embedding,      # Raw embedding of current student [B, Cemb_s]
    student_logits,         # Logits of current student [B, N_cls]
    targets,                # Ground truth targets [B]
    teacher_models,         # List of teacher model objects
    teacher_last_conv_features_list, # List of [B, Ct, Ht, Wt] from teachers
    teacher_embeddings_list, # List of [B, Cemb_t] from teachers
    teacher_logits_list,     # List of [B, N_cls] from teachers
    criterion_kd_logits_unreduced, # DistillKL(reduction='none') or similar
    student_feat_transformer, # The TransFeat module for current student
    args, device):
    """
    Prepares the state input for the PolicyTrans RL agent for a specific student.
    Output: A tuple (list_of_teacher_specific_info_tensors, teacher_ce_losses, student_teacher_logit_divs, student_teacher_feat_sims)
            as expected by PolicyTrans.
    """
    num_teachers = len(teacher_models)
    batch_size = student_logits.size(0)

    # 1. Student's features transformed to align with each teacher's last conv features
    #    student_feat_transformer(student_last_conv_feat) returns a list of tensors.
    s_transformed_conv_feats_for_teachers = student_feat_transformer(student_last_conv_feat)


    agent_teacher_specific_inputs = []
    all_teacher_ce_losses = torch.zeros(batch_size, num_teachers, device=device)
    all_student_teacher_logit_divs = torch.zeros(batch_size, num_teachers, device=device)
    all_student_teacher_feat_sims = torch.zeros(batch_size, num_teachers, device=device) # Using similarity here

    with torch.no_grad(): # Calculations for agent state should not have gradients
        for t_idx in range(num_teachers):
            t_conv_feat = teacher_last_conv_features_list[t_idx] # [B, Ct, Ht, Wt]
            t_emb = teacher_embeddings_list[t_idx]             # [B, Cemb_t] (possibly already flat)
            t_logits = teacher_logits_list[t_idx]            # [B, N_cls]
            
            s_transformed_conv_feat_for_this_teacher = s_transformed_conv_feats_for_teachers[t_idx]

            # Metric 1: Student-Teacher Feature Similarity (e.g., cosine sim on pooled features)
            # Pool both to vectors before cosine similarity
            s_vec = F.adaptive_avg_pool2d(s_transformed_conv_feat_for_this_teacher, 1).view(batch_size, -1)
            t_vec = F.adaptive_avg_pool2d(t_conv_feat, 1).view(batch_size, -1)
            feat_sim = F.cosine_similarity(s_vec, t_vec, dim=1) # [B]
            all_student_teacher_feat_sims[:, t_idx] = feat_sim

            # Metric 2: Student-Teacher Logit Divergence (KL)
            # criterion_kd_logits_unreduced should be DistillKL with reduction='none' then sum(-1)
            logit_div = criterion_kd_logits_unreduced(student_logits, t_logits) # [B]
            all_student_teacher_logit_divs[:, t_idx] = logit_div
            
            # Metric 3: Teacher's CE loss on current batch (how good is this teacher on this data)
            teacher_ce = F.cross_entropy(t_logits, targets, reduction='none') # [B]
            all_teacher_ce_losses[:, t_idx] = teacher_ce

            # Agent input per teacher: student_emb_for_agent, teacher_logits, teacher_CE, student_teacher_logit_KL, student_teacher_feat_similarity
            t_emb_flat = t_emb.view(batch_size, -1)
            
            current_teacher_info = torch.cat([
                t_emb_flat,         # Teacher's own embedding
                t_logits,           # Teacher's own logits
                feat_sim.view(batch_size, 1),
                logit_div.expand(batch_size, 1),
                teacher_ce.view(batch_size, 1)
            ], dim=1)
            agent_teacher_specific_inputs.append(current_teacher_info)
            
    return (agent_teacher_specific_inputs, all_teacher_ce_losses, all_student_teacher_logit_divs, all_student_teacher_feat_sims)


def train_rl_agent_step(student_agent_states_in_batch, # List of states (one per student for this batch from prepare_agent_state)
                        student_rewards_in_batch,    # List of reward tensors [B] (one per student for this batch)
                        rl_agent_model, agent_optimizer, args, epoch):
    """Performs a training step for the RL agent."""
    rl_agent_model.train()
    agent_optimizer.zero_grad()
    
    total_agent_loss_this_step = 0
    num_experiences = 0

    # If PolicyTrans makes one decision for all students based on one state (e.g. first student's state)
    # then this loop isn't quite right.
    # Assume for now, agent is trained on each student's experience independently contributing to gradient.
    for s_idx in range(len(student_agent_states_in_batch)):
        agent_state = student_agent_states_in_batch[s_idx] # State for student s_idx
        rewards = student_rewards_in_batch[s_idx]          # Rewards for student s_idx actions [B]
        
        batch_size = rewards.size(0)
        if batch_size == 0: continue

        # Agent's current policy (actions/weights) for this state
        # agent_state is (list_of_teacher_info, teacher_ces, student_teacher_logit_divs, student_teacher_feat_sims)
        logit_policy_weights, feat_policy_weights = rl_agent_model(agent_state) # [B, Num_Teachers]

        # BCE Loss: Encourage actions (weights close to 1) that led to high (positive) reward.
        # Target is 1. Loss is weighted by reward.
        # Rewards need to be shaped [B, 1] to multiply with per-teacher loss terms if BCE reduction is 'none'.
        # If BCE reduction is 'mean', then apply reward weighting carefully.
        # Let's use reduction='none' for BCE and then apply reward and mean.
        
        loss_logit_policy_items = F.binary_cross_entropy(logit_policy_weights, 
                                                       torch.ones_like(logit_policy_weights), 
                                                       reduction='none') # [B, Num_Teachers]
        # Weighted sum over teachers, then mean over batch
        loss_logit_policy = (loss_logit_policy_items * rewards.unsqueeze(1)).mean() 

        loss_feat_policy_items = F.binary_cross_entropy(feat_policy_weights,
                                                      torch.ones_like(feat_policy_weights),
                                                      reduction='none') # [B, Num_Teachers]
        loss_feat_policy = (loss_feat_policy_items * rewards.unsqueeze(1)).mean()

        current_agent_loss = loss_logit_policy + loss_feat_policy
        
        # Accumulate gradients if training over multiple students' experiences before optimizer.step()
        # For DDP, ensure loss is scaled by world_size if averaging gradients.
        # Or, just backward per experience. Let's do backward per experience.
        current_agent_loss.backward() # Accumulates gradients in agent_optimizer.param_groups
        
        total_agent_loss_this_step += current_agent_loss.item() * batch_size # To get average later
        num_experiences += batch_size

    if num_experiences > 0:
        # Optional: Gradient clipping for agent
        # torch.nn.utils.clip_grad_norm_(rl_agent_model.parameters(), max_norm=1.0)
        agent_optimizer.step() # Apply accumulated gradients
        
        avg_agent_loss_this_step = total_agent_loss_this_step / num_experiences
        if args.rank == 0:
             args.logger.info(f"Epoch {epoch+1} Agent Training Step: Avg Loss: {avg_agent_loss_this_step:.5f}")
    # else:
    #     if args.rank == 0:
    #         args.logger.info(f"Epoch {epoch+1} Agent Training Step: No experiences to train on.")


def train_multi_student_rl_loop(
    train_loader, student_models, teacher_models, rl_agent,
    student_feat_trans_list, student_optimizers, agent_optimizer,
    criterion_ce, criterion_kd_logits, criterion_teacher_feat, criterion_mutual_feat, # Added mutual_feat
    epoch, device, args
):
    logger = args.logger
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    
    num_students = len(student_models)
    losses_meters = [[AverageMeter(f'S{s_idx}_TotalLoss', ':.3f'),
                      AverageMeter(f'S{s_idx}_CE', ':.3f'),
                      AverageMeter(f'S{s_idx}_TkdL', ':.3f'), # Teacher Logit KD
                      AverageMeter(f'S{s_idx}_TkdF', ':.3f'), # Teacher Feat KD
                      AverageMeter(f'S{s_idx}_MkdL', ':.3f'), # Mutual Logit KD
                      AverageMeter(f'S{s_idx}_MkdF', ':.3f')] # Mutual Feat KD
                     for s_idx in range(num_students)]
    top1_meters = [AverageMeter(f'S{s_idx}_Acc@1', ':6.2f') for s_idx in range(num_students)]

    # For RL Agent training data collection
    batch_agent_states_collected = [] # List to store states for each student from this batch
    batch_agent_rewards_collected = []# List to store rewards for each student from this batch

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.size(0)

        # --- Get Teacher Outputs (once per batch) ---
        t_logits_list = []
        t_last_conv_feats_list = []
        t_embeddings_list = []
        with torch.no_grad():
            for t_model in teacher_models:
                t_feats_all, t_logits = t_model(inputs, is_feat=True)
                t_logits_list.append(t_logits.detach())
                t_last_conv_feats_list.append(t_feats_all[-2].detach()) # Usually the one for TransFeat
                t_embeddings_list.append(t_feats_all[-1].detach())     # Usually for agent state

        # --- Get All Student Outputs (for mutual learning, detached) ---
        s_logits_for_mutual_list = [None] * num_students
        s_last_conv_feats_for_mutual_list = [None] * num_students # Raw features[-2]
        
        # Temporarily set models to eval for consistent "teaching" representations if BN is involved
        # This is a nuanced point: if BN running stats affect the "teaching" signal, eval mode is better.
        # However, the student itself is training. Let's keep them in train() and use .detach().
        for s_idx_eval, s_model_eval_ddp in enumerate(student_models):
            s_model_eval = s_model_eval_ddp.module if args.distributed else s_model_eval_ddp
            # s_model_eval.eval() # Optional: Consider impact of BN
            with torch.no_grad():
                s_feats_all_eval, s_logits_eval = s_model_eval(inputs, is_feat=True)
            s_logits_for_mutual_list[s_idx_eval] = s_logits_eval.detach()
            s_last_conv_feats_for_mutual_list[s_idx_eval] = s_feats_all_eval[-2].detach()
            # s_model_eval.train() # Set back

        # --- Train each student model ---
        current_batch_student_states_for_agent = [] # Store states from this batch for agent training
        current_batch_student_rewards_for_agent = [] # Store rewards from this batch for agent training

        for s_idx in range(num_students):
            s_model_ddp = student_models[s_idx]
            s_feat_trans_ddp = student_feat_trans_list[s_idx]
            s_optimizer = student_optimizers[s_idx]

            s_model_ddp.train()
            s_feat_trans_ddp.train()
            s_optimizer.zero_grad()

            # Forward pass for current student (gradient enabled)
            s_feats_all, s_logits = s_model_ddp(inputs, is_feat=True)
            s_last_conv_feat = s_feats_all[-2] # For TransFeat and mutual feature learning
            s_embedding = s_feats_all[-1]      # For agent state

            # 1. Cross-Entropy Loss
            loss_ce = args.ce_weight * criterion_ce(s_logits, targets)
            
            # 2. Distillation from Teachers
            loss_teacher_kd_logits_total = torch.tensor(0.0, device=device)
            loss_teacher_feat_total = torch.tensor(0.0, device=device)

            if teacher_models:
                rl_agent.eval() # Agent generates weights in eval mode
                
                # Transformed student features for each teacher
                s_transformed_conv_feats = s_feat_trans_ddp(s_last_conv_feat) # List of tensors

                agent_state = prepare_agent_state_for_student(
                    s_idx, s_last_conv_feat, s_embedding, s_logits, targets,
                    teacher_models, t_last_conv_feats_list, t_embeddings_list, t_logits_list,
                    criterion_kd_logits, # Pass the original DistillKL, unreduce happens inside prepare_agent_state
                    s_feat_trans_ddp.module if args.distributed else s_feat_trans_ddp, # Pass raw TransFeat
                    args, device
                )
                current_batch_student_states_for_agent.append(agent_state) # Collect for agent training

                with torch.no_grad():
                    # Get weights from RL agent: shape [BatchSize, NumTeachers]
                    agent_logit_weights, agent_feat_weights = rl_agent(agent_state)

                for t_idx in range(len(teacher_models)):
                    # Logit KD loss from teacher t_idx
                    # criterion_kd_logits is DistillKL with reduction='batchmean' or 'mean'
                    _loss_kdl = criterion_kd_logits(s_logits, t_logits_list[t_idx]) # Scalar
                    loss_teacher_kd_logits_total += (_loss_kdl * agent_logit_weights[:, t_idx].mean()) # Weight by avg weight for this teacher

                    # Feature KD loss from teacher t_idx
                    # s_transformed_conv_feats[t_idx] is student's feature for this teacher
                    # t_last_conv_feats_list[t_idx] is this teacher's feature
                    _loss_feat = criterion_teacher_feat(s_transformed_conv_feats[t_idx], t_last_conv_feats_list[t_idx]) # Scalar
                    loss_teacher_feat_total += (_loss_feat * agent_feat_weights[:, t_idx].mean())
            
            loss_teacher_kd_logits_w = args.teacher_kd_weight * loss_teacher_kd_logits_total
            loss_teacher_feat_w = args.teacher_feat_weight * loss_teacher_feat_total
            
            # 3. Mutual Learning from Peers
            loss_mutual_kd_logits_total = torch.tensor(0.0, device=device)
            loss_mutual_feat_total = torch.tensor(0.0, device=device)
            num_valid_peers = 0
            if num_students > 1:
                for p_idx in range(num_students):
                    if s_idx == p_idx:
                        continue
                    num_valid_peers += 1
                    
                    # Mutual Logit KD
                    peer_logits_detached = s_logits_for_mutual_list[p_idx]
                    loss_mutual_kd_logits_total += criterion_kd_logits(s_logits, peer_logits_detached)
                    
                    # Mutual Feature KD (simplified)
                    peer_feat_detached = s_last_conv_feats_for_mutual_list[p_idx]
                    if s_last_conv_feat.shape == peer_feat_detached.shape: # Direct comparison if shapes match
                        loss_mutual_feat_total += criterion_mutual_feat(s_last_conv_feat, peer_feat_detached)
                
                if num_valid_peers > 0:
                    loss_mutual_kd_logits_total /= num_valid_peers
                    loss_mutual_feat_total /= num_valid_peers # Careful if no valid peers for features
            
            loss_mutual_kd_logits_w = args.mutual_kd_weight * loss_mutual_kd_logits_total
            loss_mutual_feat_w = args.mutual_feat_weight * loss_mutual_feat_total

            # --- Total Loss for Student s_idx ---
            total_loss_s = loss_ce + loss_teacher_kd_logits_w + loss_teacher_feat_w + \
                           loss_mutual_kd_logits_w + loss_mutual_feat_w
            
            total_loss_s.backward()
            s_optimizer.step()

            # Calculate reward for the RL agent based on this student's CE and Teacher-KD performance
            # Reward = -(CE_loss + Teacher_Logit_KD_loss + Teacher_Feature_KD_loss)
            # Negative losses, so higher (less negative) is better.
            # Reward should be per batch item, so use .item() carefully or ensure losses are per item.
            # Let's use scalar losses for reward.
            current_s_reward_scalar = -(loss_ce.item() + 
                                      loss_teacher_kd_logits_w.item() + 
                                      loss_teacher_feat_w.item())
            # Create a tensor of this reward for each item in the batch
            current_s_reward_tensor = torch.full((batch_size,), current_s_reward_scalar, device=device, dtype=torch.float32)
            
            # Normalize reward before storing (example: (r - mean)/std, then clamp to [0,1])
            # This normalization should ideally be done over a larger set of rewards,
            # but for simplicity, can be done per batch or per agent step.
            # For now, let's store raw (negative loss sum) rewards and normalize before agent training.
            current_batch_student_rewards_for_agent.append(current_s_reward_tensor)


            # Update loss meters for student s_idx
            losses_meters[s_idx][0].update(total_loss_s.item(), batch_size)
            losses_meters[s_idx][1].update(loss_ce.item(), batch_size)
            losses_meters[s_idx][2].update(loss_teacher_kd_logits_w.item(), batch_size)
            losses_meters[s_idx][3].update(loss_teacher_feat_w.item(), batch_size)
            losses_meters[s_idx][4].update(loss_mutual_kd_logits_w.item() if isinstance(loss_mutual_kd_logits_w, torch.Tensor) else loss_mutual_kd_logits_w, batch_size)
            losses_meters[s_idx][5].update(loss_mutual_feat_w.item() if isinstance(loss_mutual_feat_w, torch.Tensor) else loss_mutual_feat_w, batch_size)

            acc1, _ = correct_num(s_logits.detach(), targets, topk=(1, 5))
            top1_meters[s_idx].update(acc1.item() * 100.0 / batch_size, batch_size)
        
        # --- RL Agent Training Step ---
        if teacher_models and (batch_idx + 1) % args.agent_step_freq == 0:
            if current_batch_student_states_for_agent and current_batch_student_rewards_for_agent:
                # Normalize rewards across all students' experiences in this collection period
                all_rewards_flat = torch.cat(current_batch_student_rewards_for_agent) # Concatenate rewards from all students
                if all_rewards_flat.numel() > 1 : # Need at least 2 elements for std
                    rewards_mean = all_rewards_flat.mean()
                    rewards_std = all_rewards_flat.std() + 1e-7 # Add epsilon for stability
                    normalized_rewards_list = [(r_tensor - rewards_mean) / rewards_std for r_tensor in current_batch_student_rewards_for_agent]
                    # Clamp normalized rewards (e.g., to [0, 1] if agent loss expects this, or handle in agent loss)
                    # Original code clamped to [0,1]. This can be aggressive.
                    # For now, let's assume train_rl_agent_step can handle potentially negative rewards or a range.
                    # If using BCE with reward as weight, positive rewards are needed.
                    # Let's try a different normalization: shift to be mostly positive, e.g., softmax scaling or simple shift.
                    # Or, keep original clamping for consistency if PolicyTrans expects it.
                    clamped_rewards_list = [torch.clamp(nr, min=0, max=1) for nr in normalized_rewards_list]

                else: # single reward value, or single student
                    clamped_rewards_list = [torch.clamp(r_tensor, min=0, max=1) for r_tensor in current_batch_student_rewards_for_agent]


                train_rl_agent_step(current_batch_student_states_for_agent, 
                                    clamped_rewards_list, # Pass normalized & clamped rewards
                                    rl_agent, agent_optimizer, args, epoch)
            
            current_batch_student_states_for_agent = [] # Clear for next agent training cycle
            current_batch_student_rewards_for_agent = []


        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx + 1) % args.print_freq == 0 and args.rank == 0:
            log_str = f'Epoch: [{epoch+1}][{batch_idx+1}/{len(train_loader)}]\t' \
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\n'
            for s_idx_log in range(num_students):
                arch_name = args.student_arch_list[s_idx_log]
                log_str += (f"  S{s_idx_log}({arch_name}): "
                            f"Loss {losses_meters[s_idx_log][0].avg:.3f} "
                            f"(CE {losses_meters[s_idx_log][1].avg:.2f}|"
                            f"TLogit {losses_meters[s_idx_log][2].avg:.2f}|"
                            f"TFeat {losses_meters[s_idx_log][3].avg:.2f}|"
                            f"MLogit {losses_meters[s_idx_log][4].avg:.2f}|"
                            f"MFeat {losses_meters[s_idx_log][5].avg:.2f}) "
                            f"Acc@1 {top1_meters[s_idx_log].avg:.2f}%\n")
            logger.info(log_str.strip())
            # Reset meters for next print interval
            for s_meters in losses_meters:
                for meter in s_meters: meter.reset()
            for meter in top1_meters: meter.reset()
            batch_time.reset(); data_time.reset()


def test_multi_student_rl_loop(val_loader, student_models, criterion_ce, epoch, device, args):
    logger = args.logger
    num_students = len(student_models)
    batch_time = AverageMeter('Time', ':6.3f')
    
    student_val_losses = [AverageMeter(f'S{s_idx}_ValLoss', ':.4e') for s_idx in range(num_students)]
    student_val_top1 = [AverageMeter(f'S{s_idx}_ValAcc@1', ':6.2f') for s_idx in range(num_students)]
    
    for s_model_ddp in student_models:
        s_model_ddp.eval()

    all_student_accuracies = [0.0] * num_students

    with torch.no_grad():
        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = inputs.size(0)

            for s_idx, s_model_ddp in enumerate(student_models):
                s_model = s_model_ddp.module if args.distributed else s_model_ddp
                
                # We only need logits for validation
                _, s_logits = s_model(inputs, is_feat=True) 
                
                loss = criterion_ce(s_logits, targets)
                student_val_losses[s_idx].update(loss.item(), batch_size)

                acc1, _ = correct_num(s_logits, targets, topk=(1,5))
                student_val_top1[s_idx].update(acc1.item() * 100.0 / batch_size, batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % args.print_freq == 0 and args.rank == 0:
                log_str = f'Test Epoch: [{epoch+1}][{batch_idx+1}/{len(val_loader)}]\tTime {batch_time.avg:.3f}\n'
                for s_idx_log in range(num_students):
                    arch_name = args.student_arch_list[s_idx_log]
                    log_str += f"  S{s_idx_log}({arch_name}): ValLoss {student_val_losses[s_idx_log].avg:.4f} Acc@1 {student_val_top1[s_idx_log].avg:.2f}%\n"
                logger.info(log_str.strip())
    
    if args.rank == 0: # Log final validation results from rank 0
        logger.info(f"--- Validation Summary for Epoch {epoch+1} ---")
        for s_idx_log in range(num_students):
            all_student_accuracies[s_idx_log] = student_val_top1[s_idx_log].avg
            arch_name = args.student_arch_list[s_idx_log]
            logger.info(f"  Student {s_idx_log} ({arch_name}): Final ValLoss {student_val_losses[s_idx_log].avg:.4f}, Final Acc@1 {all_student_accuracies[s_idx_log]:.2f}%")
        logger.info("------------------------------------")
        
    # Synchronize all_student_accuracies across DDP processes if needed for consistent return value
    # For now, rank 0 has the authoritative values logged.
    # If other ranks need it, an all_gather operation would be required.
    # This function is typically called by rank 0 for saving best models, so this might be fine.
    
    return all_student_accuracies # List of average top1 accuracies


def main():
    args = parse_arguments()
    setup_ddp(args) # Initializes DDP and sets args.rank, args.world_size, args.local_rank, args.distributed
    
    args.logger = get_logger(args) # Logger setup after DDP init to use args.rank

    if args.seed is not None:
        seed_val = args.seed + args.rank # Different seed for each process
        random.seed(seed_val)
        torch.manual_seed(seed_val)
        np.random.seed(seed_val) # Also seed numpy if used by dataloaders/transforms
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
        cudnn.deterministic = True
        cudnn.benchmark = False # Deterministic usually means benchmark = False
        args.logger.warning('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                            'which can slow down your training considerably! ')
    else:
        cudnn.benchmark = True # Good for performance if input sizes don't change

    try:
        main_worker(args)
    finally:
        cleanup_ddp()

if __name__ == '__main__':
    main()
