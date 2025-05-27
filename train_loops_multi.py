from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num
import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np
from distiller_zoo import FeatureKLLoss, FeatureMSELoss
from loguru import logger


def get_actions(agent_pred):
    batch_size = agent_pred.size(0)
    teacher_num = agent_pred.size(1)
    index = torch.from_numpy(np.random.randint(0, teacher_num, batch_size).astype(np.int64)).cuda(args.gpu)
    random_select =  F.one_hot(index, num_classes=teacher_num).float().cuda(args.gpu)
    actions = torch.where(agent_pred>=0.5, torch.ones_like(agent_pred), torch.zeros_like(agent_pred)).cuda(args.gpu)
    is_random = (actions.sum(1) ==  0)[:, None].float().cuda(args.gpu)
    actions = actions + random_select * is_random
    return actions


def train_agent(args, epoch, agent_state, agent_rewards, logits_agent_actions, agent, agent_optimizer):
    agent.train()
    agent_loss = AverageMeter('agent_loss', ':.4e')

    for state, rewards, actions in zip(agent_state, agent_rewards, logits_agent_actions):
        
        agent_pred = agent(state)
        agent_optimizer.zero_grad() 
        
        action_label = torch.ones_like(agent_pred[0]).detach()
        loss_logits = F.binary_cross_entropy(agent_pred[0], action_label, weight=rewards.unsqueeze(-1))
        loss_feature = F.binary_cross_entropy(agent_pred[1], action_label, weight=rewards.unsqueeze(-1))
        loss = loss_feature + loss_logits
        loss.backward()
        agent_optimizer.step()

        agent_loss.update(loss.item(), actions.size(0))

    if args.rank == 0:
        logger.info('Epoch:{}, agent Loss:{:.6f}'.format(epoch, agent_loss.avg))

def get_agent_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div):
    trans_student_embeddings = []
    for idx in range(len(trans_student_features)):
        trans_student_embedding = F.adaptive_avg_pool2d(trans_student_features[idx], (1,1))
        trans_student_embedding = trans_student_embedding.view(trans_student_embedding.size(0), -1)
        trans_student_embeddings.append(trans_student_embedding)

    teacher_infos = []
    t_ces = []
    t_s_feat_div = []
    t_s_logit_div = []
    for idx in range(len(teacher_embeddings)):            
        feat_cos_sim = F.cosine_similarity(trans_student_embeddings[idx], teacher_embeddings[idx]).unsqueeze(-1)
        t_s_feat_div.append(feat_cos_sim)
        logit_kl = criterion_div(logits, teacher_logits[idx], unreduce=True).unsqueeze(-1)
        t_s_logit_div.append(logit_kl)
        teachers_ce = F.cross_entropy(teacher_logits[idx], targets, reduction='none').unsqueeze(-1)
        t_ces.append(teachers_ce)
        teacher_info = torch.cat([feat_cos_sim, logit_kl, teachers_ce, teacher_embeddings[idx], teacher_logits[idx]], dim=1).detach()
        teacher_infos.append(teacher_info)
    t_ces = torch.cat(t_ces, dim=1).detach()
    t_s_logit_div = torch.cat(t_s_logit_div, dim=1).detach()
    t_s_feat_div = torch.cat(t_s_feat_div, dim=1).detach()
    return teacher_infos, t_ces, t_s_logit_div, t_s_feat_div



def train_multi_students(train_loader, models, criterion_list, optimizer, epoch, device, 
          args, agents, feat_trans_list, teacher_models, agent_optimizer):
    """
    Train multiple student models together
    """
    
    train_losses = [AverageMeter('train_loss', ':.4e') for _ in models]
    train_losses_cls = [AverageMeter('train_loss_cls', ':.4e') for _ in models]
    train_losses_kd = [AverageMeter('train_loss_kd', ':.4e') for _ in models]
    train_losses_feat = [AverageMeter('train_loss_feat', ':.4e') for _ in models]

    top1_nums = [0] * len(models)
    top5_nums = [0] * len(models)
    total = 0
    # lrs = [0] * len(models)
    # for optimizer_idx, optimizer in enumerate(optimizers):
    #     lrs[optimizer_idx] = adjust_lr(optimizer, epoch, args)  # assume same lr for all students
    lr = adjust_lr(optimizer, epoch, args)
    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]

    for model in models:
        model.train()
    for agent in agents:
        agent.eval()
        
    # Store agent data for all students
    all_agent_states = [[] for _ in models]
    all_logits_agent_actions = [[] for _ in models]
    all_feature_agent_actions = [[] for _ in models]
    all_agent_rewards = [[] for _ in models]
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_start_time = time.time()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Zero gradients for all students
        # for optimizer in optimizers:
        optimizer.zero_grad()
        
        # Get teacher features once for all students
        teacher_logits = []
        teacher_features = []
        teacher_embeddings = []
        with torch.no_grad():
            for t_model in teacher_models:
                t_features, t_logits = t_model(inputs, is_feat=True)
                t_feature = t_features[-1]
                t_feature = t_feature.detach() 
                t_logits = t_logits.detach() 
                
                teacher_features.append(t_features[-2])
                teacher_logits.append(t_logits)
                teacher_embeddings.append(t_features[-1])

        total_loss = 0
        
        # Process each student model
        for student_idx, (model, agent, feat_trans) in enumerate(zip(models, agents, feat_trans_list)):
            
            features, logits = model(inputs, is_feat=True) 
            trans_student_features = feat_trans(features[-2])
            
            agent_state = get_agent_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div)
            all_agent_states[student_idx].append(agent_state)
            
            with torch.no_grad():
                logits_actions, feature_actions = agent(agent_state)
            if epoch == 0:
                logits_actions = torch.ones_like(logits_actions).cuda(args.gpu)
                feature_actions = torch.ones_like(feature_actions).cuda(args.gpu)
            logits_actions = logits_actions.detach()
            feature_actions = feature_actions.detach()

            all_logits_agent_actions[student_idx].append(logits_actions)
            all_feature_agent_actions[student_idx].append(feature_actions)

            if args.rank == 0 and batch_idx % 10 == 0 :  # Only log for first student to reduce output
                logger.info('Student {}, actions:{}'.format(student_idx, str(logits_actions[0])))
   
                
            loss_cls = criterion_ce(logits, targets)
            
            loss_kd = torch.tensor(0.).cuda(args.gpu)
            for idx in range(len(teacher_models)):
                loss_kd = loss_kd + (logits_actions[:, idx] * criterion_div(logits, teacher_logits[idx].detach(), unreduce=True)).mean()
                
            loss_feat = torch.tensor(0.).cuda(args.gpu)
            
            if args.feat_kd == 'mse':
                feat_kd_func = FeatureMSELoss()
            elif args.feat_kd == 'kl':
                feat_kd_func = FeatureKLLoss(args.kd_T)

            for idx in range(len(teacher_models)):
                loss_feat = loss_feat + (feature_actions[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx])).mean()

            loss_feat = args.feat_weight * loss_feat
            
            loss = loss_cls + loss_kd + loss_feat
            total_loss += loss  # Accumulate loss from all students
            
            # Calculate rewards for this student
            sample_ce_loss = F.cross_entropy(logits, targets, reduction='none')
            sample_kd_loss = torch.tensor(0.).cuda(args.gpu)
            sample_feat_loss = torch.tensor(0.).cuda(args.gpu)
            for idx in range(len(teacher_models)):
                sample_kd_loss = sample_kd_loss + logits_actions[:, idx] * criterion_div(logits, teacher_logits[idx].detach(), unreduce=True)
                sample_feat_loss = sample_feat_loss + (feature_actions[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx]))
            reward = -(sample_ce_loss + sample_kd_loss+ args.feat_weight * sample_feat_loss)
            rewards_mean = reward.mean()
            rewards_std = reward.std()
            normalized_reward = (reward - rewards_mean) / rewards_std
            normalized_reward = normalized_reward.detach()
            normalized_reward = torch.clamp(normalized_reward, min=0, max=1)
            all_agent_rewards[student_idx].append(normalized_reward)
            
            # Update metrics for this student
            train_losses[student_idx].update(loss.item(), inputs.size(0))
            train_losses_cls[student_idx].update(loss_cls.item(), inputs.size(0))
            train_losses_kd[student_idx].update(loss_kd.item(), inputs.size(0))
            train_losses_feat[student_idx].update(loss_feat.item(), inputs.size(0))
            
            top1, top5 = correct_num(logits, targets, topk=(1, 5))
            top1_nums[student_idx] += top1
            top5_nums[student_idx] += top5
        
        # Backward pass for combined loss
        total_loss.backward()
        
        # Update all optimizers
        # for optimizer in optimizers:
        optimizer.step()
            
        total += targets.size(0)

        if args.rank == 0:
            avg_cls_loss = sum(train_losses_cls[i].avg for i in range(len(models))) / len(models)
            avg_kd_loss = sum(train_losses_kd[i].avg for i in range(len(models))) / len(models)
            avg_feat_loss = sum(train_losses_feat[i].avg for i in range(len(models))) / len(models)
            avg_acc = sum((top1_nums[i]/total*100.).item() for i in range(len(models))) / len(models)
            
            logger.info('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Avg CLS Loss:{:.2f},' 
                'Avg KD Loss:{:.2f}, Avg Feature Loss:{:.2f}, Avg Top-1 Acc:{:.2f}'.format(
                epoch, batch_idx, len(train_loader), lr, time.time()-batch_start_time, 
                avg_cls_loss, avg_kd_loss, avg_feat_loss, avg_acc))
                
        # Train agents periodically
        if batch_idx % args.agent_step == 0 and batch_idx != 0:
            for student_idx, agent in enumerate(agents):
                train_agent(args, epoch, all_agent_states[student_idx], all_agent_rewards[student_idx], 
                           all_logits_agent_actions[student_idx], agent, agent_optimizer)
            # Clear accumulated data
            all_agent_states = [[] for _ in models]
            all_logits_agent_actions = [[] for _ in models]
            all_feature_agent_actions = [[] for _ in models]
            all_agent_rewards = [[] for _ in models]
    
    # Calculate final accuracies
    accs1 = [top1_nums[i] / total for i in range(len(models))]
    accs5 = [top5_nums[i] / total for i in range(len(models))]

    if args.rank == 0:
        for student_idx in range(len(models)):
            logger.info('Student {} - Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                        '\n Train_loss:{:.5f}'
                        '\t Train_loss_cls:{:.5f}'
                        '\t Train_loss_kd:{:.5f}'
                        '\t Train_loss_feat:{:.5f}'
                        '\nTrain top-1 accuracy:{:.2f}'
                        .format(student_idx, epoch, lr, time.time() - start_time,
                                train_losses[student_idx].avg,
                                train_losses_cls[student_idx].avg,
                                train_losses_kd[student_idx].avg,
                                train_losses_feat[student_idx].avg,
                                accs1[student_idx]*100.))
                                
    # Train remaining agent data if any
    for student_idx, agent in enumerate(agents):
        if len(all_agent_states[student_idx]) > 0:
            train_agent(args, epoch, all_agent_states[student_idx], all_agent_rewards[student_idx], 
                       all_logits_agent_actions[student_idx], agent, agent_optimizer)


def test_multi_students(epoch, models, device, val_loader, criterion_ce, args, verbose=True):
    """
    Test multiple student models
    """
    test_losses_cls = [AverageMeter('Loss', ':.4e') for _ in models]
    top1_nums = [0] * len(models)
    top5_nums = [0] * len(models)
    total = 0
    
    for model in models:
        model.eval()
        
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            batch_start_time = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            for student_idx, model in enumerate(models):
                features, logits = model(inputs, is_feat=True)
                loss_cls = criterion_ce(logits, targets)
                test_losses_cls[student_idx].update(loss_cls.item(), inputs.size(0))

                top1, top5 = correct_num(logits, targets, topk=(1, 5))
                top1_nums[student_idx] += top1
                top5_nums[student_idx] += top5
            
            total += targets.size(0)
            
            if args.rank == 0 and verbose and batch_idx % 50 == 0:
                avg_acc = sum((top1_nums[i]/total*100.).item() for i in range(len(models))) / len(models)
                logger.info('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Avg Test Top-1 Acc:{:.4f}'.format(
                    epoch, batch_idx, len(val_loader), time.time()-batch_start_time, avg_acc))
                    
        class_accs1 = [round((top1_nums[i]/total*100.).item(), 4) for i in range(len(models))]
        class_accs5 = [round((top5_nums[i]/total*100.).item(), 4) for i in range(len(models))]

        if args.rank == 0 and verbose:
            for student_idx in range(len(models)):
                logger.info('Student {} - Test epoch:{}\t Test_loss_cls:{:.5f}\nTest top-1 accuracy: {}\nTest top-5 accuracy: {}'
                            .format(student_idx, epoch, test_losses_cls[student_idx].avg, 
                                   str(class_accs1[student_idx]), str(class_accs5[student_idx])))
    return class_accs1


