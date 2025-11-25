import pytz
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *
from torch.optim import Adam, AdamW, lr_scheduler
from datetime import datetime
from sklearn.preprocessing import label_binarize


class WarmUpLR(lr_scheduler._LRScheduler):
    """warmup_training learning rate scheduler

    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.warmup_epochs = warmup_epochs
        self.init_lr = base_lr/self.warmup_epochs
        super().__init__(optimizer)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [self.init_lr * epoch for epoch in range(1,self.warmup_epochs+1)]


# def cal_scores(logits, labels, num_classes):
#     logits = torch.tensor(logits)
#     labels = torch.tensor(labels)
#     predicted_classes = torch.argmax(logits, dim=1)
#     accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())
#     probs = F.softmax(logits, dim=1)
#     if num_classes > 2:
#         macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='macro', multi_class='ovr')
#         micro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='micro', multi_class='ovr')
#         weighted_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='weighted', multi_class='ovr')
#     else:
#         macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())
#         weighted_auc = micro_auc = macro_auc
#     weighted_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
#     weighted_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
#     weighted_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
#     macro_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='macro')
#     macro_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='macro')
#     macro_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='macro')
#     micro_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='micro')
#     micro_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='micro')
#     micro_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='micro')
#     baccuracy = balanced_accuracy_score(labels.numpy(), predicted_classes.numpy())
#
#     metrics = {'acc': accuracy,  'bacc': baccuracy,
#                'macro_auc': macro_auc, 'micro_auc': micro_auc, 'weighted_auc':weighted_auc,
#                'macro_f1': macro_f1, 'micro_f1': micro_f1, 'weighted_f1': weighted_f1,
#                'macro_recall': macro_recall, 'micro_recall': micro_recall,'weighted_recall': weighted_recall,
#                'macro_pre': macro_precision, 'micro_pre': micro_precision,'weighted_pre': weighted_precision,
#                }
#
#     return metrics


def cal_scores(logits, labels):
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    y_hats = torch.argmax(logits, dim=1, keepdim=True)
    n_class = len(set(logits[0]))
    labels = labels.reshape(-1).detach().cpu().numpy()
    y_hats = y_hats.reshape(-1).detach().cpu().numpy()

    # fpr, tpr, thresholds = roc_curve(labels, y_hats)
    # bag_auc = auc(fpr, tpr)
    # threshold = thresholds[np.argmax(tpr - fpr)]
    bag_acc = accuracy_score(labels, y_hats)
    bag_f1 = f1_score(labels, y_hats, average='macro')

    labels = label_binarize(labels, classes=np.arange(n_class))  # 根据类别数调整
    y_hats = label_binarize(y_hats, classes=np.arange(n_class))
    bag_auc = roc_auc_score(labels, y_hats, multi_class="ovr", average="macro")

    metrics = {"acc": bag_acc, "f1": bag_f1, "auc": bag_auc}

    return metrics


def get_act(act):
    if act.lower() == 'relu':
        return nn.ReLU()
    elif act.lower() == 'gelu':
        return nn.GELU()
    elif act.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    elif act.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif act.lower() == 'tanh':
        return nn.Tanh()
    elif act.lower() == 'silu':
        return nn.SiLU()
    else:
        raise Exception(f'Invalid activation function: {act}')


def get_criterion(criterion):
    if criterion == 'ce':
        return nn.CrossEntropyLoss()
    else:
        raise Exception("No such loss")


def get_scheduler(args, optimizer, base_lr):
    sch = args.model.scheduler.which
    warmup = args.model.scheduler.warmup
    warmup_scheduler = WarmUpLR(optimizer, warmup,base_lr)
    if sch == 'step':
        step_size = args.model.scheduler.step_config.step_size
        gamma = args.model.scheduler.step_config.gamma
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler,warmup_scheduler
    elif sch == 'multi_step':
        milestones = args.model.scheduler.multi_step_config.milestones
        gamma = args.model.scheduler.multi_step_config.gamma
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        return scheduler,warmup_scheduler
    elif sch == 'exponential':
        gamma = args.model.scheduler.exponential_config.gamma
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        return scheduler, warmup_scheduler
    elif sch == 'cosine':
        T_max = args.model.scheduler.cosine_config.T_max
        eta_min = args.model.scheduler.cosine_config.eta_min
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        return scheduler, warmup_scheduler
    elif sch == 'none':
        return None, warmup_scheduler


def get_optimizer(args, model):
    opt = args.model.optimizer.which

    if opt == 'adam':
        lr = args.model.optimizer.adam_config.lr
        weight_decay = args.model.optimizer.adam_config.weight_decay
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    elif opt == 'adamw':
        lr = args.model.optimizer.adamw_config.lr
        weight_decay = args.model.optimizer.adamw_config.weight_decay
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    else:
        raise Exception("No such optimizer!")

    return optimizer, lr


def get_dtfd_optimizer(args, params):
    # trainable_parameters = filter(lambda p: p.requires_grad, mil_model.parameters())
    trainable_parameters = params
    opt = args.model.optimizer.which
    if opt == 'adam':
       lr = args.model.optimizer.adam_config.lr
       weight_decay = args.model.optimizer.adam_config.weight_decay
       optimizer = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=weight_decay)
       return optimizer,lr
    elif opt == 'adamw':
       lr = args.model.optimizer.adamw_config.lr
       weight_decay = args.model.optimizer.adamw_config.weight_decay
       optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
       return optimizer,lr


def get_time():

    tz = pytz.timezone('Asia/Shanghai')

    now = datetime.now(tz)

    return now.strftime("%Y-%m-%d-%H-%M")


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

