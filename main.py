import argparse
import math
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import csv
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import pickle
from lib.dataset.get_text_mask_manual import DatasetSliceTextManual
from lib.utils.metrics import calculate_multiclass_metrics, log_and_save_metrics, one_hot, calculate_net_benefit, multi_class_mAP, voc_ap
from lib.utils.plotmetrics import plot_all_experts_tsne, plot_dca_curves, plot_losses, plot_multiclass_pr_curve, plot_roc_curve, plot_tsne, plot_pr_curve
from lib.utils.logger import setup_logger
from lib.utils.loss import build_loss, Diversity_loss
import torch.nn.functional as F
from collections import OrderedDict
from lib.utils.config_ import get_raw_dict
from lib.models.resnext_MoE import ResNeXt_MoE
from lib.models.resnext import ResNeXt_Attention_Moe,ResNeXt_Attention_Moe_DCAC,ResNeXt_Cos_Moe,ResNeXt_Gating,ResNeXt_Attention_Gating_Moe, ResNeXt_Attention_Cos_Gating_Moe,ResNeXt_Attention_Gating_Text_Moe,ResNeXt_Channel_Attention_Fuse_Gating_Moe,ResNeXt_Attention_Fuse_Gating_Moe,ResNeXt_Channel_Attention_Gating_Moe
from lib.models.model import ResNeXt50Model
from lib.models.other_models import Efficient_Attention_Gating_Moe, DenseNet_Attention_Gating_MoE, vgg19_attention_gating_moe, vgg16_attention_gating_moe, resnet50_attention_gating_moe
from lib.models.other_models import densenet201_attention_gating_moe,densenet169_attention_gating_moe
os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '8854'# We use the sh file to specify the corresponding ports and DEVICES
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def parser_args():#下面是参数设置
    parser = argparse.ArgumentParser(description='Training')
    #最基础的设置，一些文件路径等
    parser.add_argument('--fold', type=int, default=3,
                        help="the current fold number for validation")
    parser.add_argument('--image_path', type=str, default=r'/data/wuhuixuan/data/padding_crop',
                        help="the location of image data")
    parser.add_argument('--fold_json', type=str, default=r'/data/huixuan/data/data_chi/TRG_patient_folds.json',
                        help="the location of the JSON file containing information about the ten-fold data split")
    parser.add_argument('--manual_csv_path', type=str, default=r'/data/wuhuixuan/code/Self_Distill_MoE/data/selected_features_22_with_id_label_fold_norm.csv',
                        help="the location of the CSV file storing extracted and filtered radiomic features")
    parser.add_argument('--csv_path', type=str, default=r'/data/huixuan/data/data_chi/label.csv',
                        help="the location of the CSV file storing prediction labels")
    parser.add_argument('--output', default='/data16t/huixuan/code/Self_Distill_MoE/out',
                        help="the output location for saving trained models and various validation plots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. Default is False.')
    parser.add_argument('--result_file', default='/data/wuhuixuan/code/Self_Distill_MoE/result/MultiExperts_ASUS_Diversity_uskd.csv',
                        help='the location of the CSV file storing training and validation results for each epoch, and the final validation results')
    parser.add_argument('--note', help='note', default='Causal experiment')
    parser.add_argument('--model_type', type=str, default='ResNeXt_Attention_Gating_Moe',
                        help='the type of model for the current experiment')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],#优化器
                        help='which optim to use')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--resume',  type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',#是不是测试模式
                        help='evaluate model on validation set')
    #可以调节的参数设置
    parser.add_argument('--criterion_type', type=str,default='ajs_uskd_mixed',
                        help="Name of the criterion to use")
    parser.add_argument('--weights', nargs='+', type=float, help='List of Loss weight')
    parser.add_argument('--focal_weight', type=float, default = 1.0, help='Focal Loss和CE Loss的比例')
    parser.add_argument('--diversity_metric', type=str, default='var',
                    help="choose which diversity metric to use")
    parser.add_argument('--lambda_v', type=float, default=0.5,
                        help="weight of the diversity loss")
    parser.add_argument('--text_dim', type=int, default=25,
                        help="dimension of the input text")
    parser.add_argument('--text_feature_dim', type=int, default=128,
                        help="feature dimension of the text feature")
    parser.add_argument('--num_class', default=2, type=int,
                        help="number of classes for model output")
    parser.add_argument('--use_criterion_total', default=True, type=bool,
                        help="whether to use the total criterion")
    parser.add_argument('--use_text', default=False, type=bool,
                        help="whether to use text (currently not added to the model)")
    parser.add_argument('--extra_input', default=True, type=bool,
                        help="whether to use radiomic data")
    parser.add_argument('--loss_ajs_weight_target', default=0.5, type=float,
                        help="target weight of ajs")
    parser.add_argument('--ajs_weight', default=0.5, type=float,
                        help="weight of ajs")
    parser.add_argument('--uskd_weight', default=0.5, type=float,
                        help="weight of uskd")
    parser.add_argument('--num_experts', default=3, type=int,
                        help="number of experts")
    parser.add_argument('--logit_method', default='ResNeXt_Attention_Gating_Moe', type=str,
                        help="name of the model for testing (previously used methods combining machine learning and deep learning for prediction)")
    parser.add_argument('--logit_alpha', default=0.5, type=float,
                        help="weight of the first method's prediction")
    parser.add_argument('--logit_beta', default=0.5, type=float,
                        help="weight of the second method's prediction")
    parser.add_argument('--logit_gamma', default=None, type=float,
                        help="weight of the third method's prediction")
    parser.add_argument('--img_size', default=224, type=int,
                        help='size of input images')
    parser.add_argument('--eps', default=1e-5, type=float,#防止为0，暂时没找到在哪用的
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--gamma', default=2.0, type=float,
                        metavar='gamma', help='gamma for focal loss')
    parser.add_argument('--alpha', default=0.25, type=float,
                        metavar='alpha', help='alpha for focal loss')
    parser.add_argument('--loss_dev', default=2, type=float,
                        help='scale factor for loss')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',#epoch
                        help='number of total epochs to run')
    parser.add_argument('--val_interval', default=1, type=int, metavar='N',#原来是1哈，就是多少个epoch之后进行验证
                        help='interval of validation')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',#是不是从头开始训练
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,#batch_size
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,#学习率我感觉可以低
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,#
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,#结果就是这个没用到每一轮都得print
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',#这个是ema model训练的参数我们在自己模型中定义了这个我们就去掉
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')
    parser.add_argument('--display_experts', type=bool,default=True,
                        help="Display every experts result or not.")
    # distribution training，下面是分布式训练的东西，还没涉及到
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', default=1024, type=int,#extra_dim
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--extra_dim', default=22, type=int,#extra_dim,临床信息的话维度是25，机器学习特征的话维度是22
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.2, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true',
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true',
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")
    # * raining，是不是提前停止等，在参数没有太大变化或者在损失爆炸的时候停止
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    
    
    args = parser.parse_args()
    return args


def get_args():
    args = parser_args()
    return args
best_mAP = 0
best_meanAUC = 0
best_f1_score = 0
def extract_features(loader, model, args):
    """
    The function `extract_features` processes data through a model, extracting features, labels, and
    predictions.
    
    :param loader: A data loader that provides batches of data for processing. It typically contains
    images, clinical features, manual features, targets, and indices for each batch
    :param model: The `model` parameter in the `extract_features` function is a neural network model
    that is used to extract features from input data. It takes input images and possibly additional
    features, such as manual features and clinic features, and produces output features and predictions.
    The model is expected to have specific components like
    :param args: The `args` parameter in the `extract_features` function is a dictionary or object
    containing various arguments or configuration settings for the feature extraction process. These
    arguments might include settings like whether to use additional input data (`extra_input`), whether
    to use text input (`use_text`), and whether to enable
    :return: The function `extract_features` returns three main outputs:
    1. `all_features`: A numpy array containing the extracted features from the model for all input
    images.
    2. `all_labels`: A numpy array containing the labels for all input images.
    3. `all_predictions`: A numpy array containing the predictions made by the model for all input
    images.
    """
    all_features = []
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for i, (images, clinic_feature, manual_features, target, idx) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            manual_features = manual_features.float().flatten(1).cuda(non_blocking=True)
            clinic_feature = clinic_feature.float().flatten(1).cuda(non_blocking=True)
            with autocast(enabled=args.amp):
                if args.extra_input:
                    extra_input=manual_features
                else:
                    extra_input=None
                if args.use_text:
                    outputs = model(images, extra_input=extra_input, text_input = clinic_feature)
                else:
                    outputs = model(images, extra_input=extra_input)
                
                features = outputs['features'].detach().cpu().numpy()
                predictions = outputs['output'].detach().cpu()
                logits = outputs['logits']
                predictions = F.softmax(predictions,dim=1)
                predictions = predictions.numpy()
            all_features.append(features)
            all_labels.append(target.detach().cpu().numpy())
            all_predictions.append(predictions)
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)#加权后的输出
    return all_features, all_labels, all_predictions 
##################################################################################
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name,
                             val=str(datetime.timedelta(seconds=int(self.val))),
                             sum=str(datetime.timedelta(seconds=int(self.sum))))


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def kill_process(filename: str, holdpid: int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True,
                                  cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist
def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict
def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger,criterion_total = None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    speed_gpu = AverageMeter('S1', ':.1f')
    speed_all = AverageMeter('SA', ':.1f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, speed_gpu, speed_all, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))#调整学习率
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    model.train()
    ema_m.eval()
    end = time.time()

    all_targets = []
    all_outputs = []  # Assuming three heads
    for i, (images, clinic_feature, manual_features, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        manual_features = manual_features.float().flatten(1).cuda(non_blocking=True)
        clinic_feature = clinic_feature.float().flatten(1).cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp):

            if args.extra_input:
                extra_input=manual_features
            else:
                extra_input=None
            if args.use_text:
                outputs = model(images, extra_input=extra_input, text_input = clinic_feature)
            else:
                outputs = model(images, extra_input=extra_input)
            loss = 0.0
            output = outputs['output']
            logits = outputs['logits']
            weights = None
            # print('weights:'+str(weights))
            prob_y = torch.sigmoid(logits)
            one_hot_target = F.one_hot(target, num_classes=args.num_class)
            prob_y_stack = torch.sigmoid(logits)
            probs_y = []
            for i in range(args.num_experts):#这个计算多次分类头输出用这个计算的多个输出进行后续计算，多样性损失
                prob_y = prob_y_stack[:,i,:]
                if args.diversity_metric == 'var':
                    prob_y_vec = torch.masked_select(input=prob_y, mask=one_hot_target.bool())#计算目标类别预测的多样性，利用方差进行衡量          
                    probs_y.append(prob_y_vec.unsqueeze(0))#这里就计算
                else:
                    loss_diversity = 0.0
            logits_list = list(torch.unbind(logits, dim=1))
            loss_t = 0
            if criterion_total:#一方面思考一下没有这个criterion_total会不会对门控机制进行监督，一方面排查一下其他损失中包不包含gating_output
                loss_t=criterion_total(output, target)
            loss = criterion(logits_list, target, weights)# output_logits, targets, extra_info=None, return_expert_losses=False# output_logits, targets, extra_info=None, return_expert_losses=False
            
            print('Total Loss:'+str(loss_t))
            if args.diversity_metric == 'erm':
                loss_diversity = 0.0
            elif args.diversity_metric == 'var':
                probs_y = torch.cat(probs_y, dim=0)
                X = torch.sqrt(torch.log(2/(1+probs_y)) + probs_y * torch.log(2*probs_y/(1+probs_y)) + 1e-6)
                loss_diversity = (X.pow(2).mean(dim=0) - X.mean(dim=0).pow(2)).mean()
            elif args.diversity_metric == 'difference':
                loss_diversity = Diversity_loss(logits, target)
            else:
                raise NotImplementedError
            loss = loss - args.lambda_v * loss_diversity + loss_t#loss4.2, loss_diversity 0.0014
            logger.info('loss: {:.3f}'.format(loss))

            if args.loss_dev > 0:
                loss *= args.loss_dev

        losses.update(loss.item(), images.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
        all_outputs.append(logits.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        lr.update(get_learning_rate(optimizer))

        if epoch > args.ema_epoch:
            ema_m.update(model)
        batch_time.update(time.time() - end)
        end = time.time()
        speed_gpu.update(images.size(0) / batch_time.val, batch_time.val)
        speed_all.update(images.size(0) * dist.get_world_size() / batch_time.val, batch_time.val)

        if i % args.print_freq == 0:
            progress.display(i, logger)

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)

    metrics, avg_metrics, avg_outputs = calculate_multiclass_metrics(all_targets, all_outputs)
    log_and_save_metrics(args=args, metrics=metrics, avg_metrics=avg_metrics, mode='train', logger=logger,epoch=epoch, loss = losses.avg)#args=args, metrics=metrics, avg_metrics=avg_metrics, mode='test', logger=logger

    return losses.avg, avg_metrics
def validate(val_loader, model, criterion, args, logger, epoch, criterion_total):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    model.eval()
    save_path = os.path.join(args.output, 'validation_results.csv')

    all_targets = []
    all_logits = []
    # extra_info = {}
    with torch.no_grad():
        end = time.time()
        loss_diversity_total = 0.0
        for i, (images, clinic_feature, manual_features, target, _) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            manual_features = manual_features.float().flatten(1).cuda(non_blocking=True)
            clinic_feature = clinic_feature.float().flatten(1).cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.extra_input:
                    extra_input=manual_features
                else:
                    extra_input=None
                if args.use_text:
                    outputs = model(images, extra_input=extra_input, text_input = clinic_feature)
                else:
                    outputs = model(images, extra_input=extra_input)
                output = outputs['output']
                logits = outputs['logits']
                weights = None
                    
                one_hot_target = F.one_hot(target, num_classes=args.num_class)
                prob_y_stack = torch.sigmoid(logits)
                probs_y = []
                for i in range(args.num_experts):#这个计算多次分类头输出用这个计算的多个输出进行后续计算
                    prob_y = prob_y_stack[:,i,:]
                    if args.diversity_metric == 'var':
                        prob_y_vec = torch.masked_select(input=prob_y, mask=one_hot_target.bool())          
                        probs_y.append(prob_y_vec.unsqueeze(0))
                    else:
                        loss_diversity = 0.0
                # features = outputs['features']
                logits_list = list(torch.unbind(logits, dim=1))
                
                loss = criterion(logits_list, target, weights)# output_logits, targets, extra_info=None, return_expert_losses=False
                loss_t = 0.0
                if criterion_total:
                    loss_t=criterion_total(output, target)
                if args.diversity_metric == 'erm':
                    loss_diversity = 0.0
                elif args.diversity_metric == 'var':
                    probs_y = torch.cat(probs_y, dim=0)
                    X = torch.sqrt(torch.log(2/(1+probs_y)) + probs_y * torch.log((2*probs_y + 1e-6)/(1+probs_y)) + 1e-6)
                    loss_diversity = (X.pow(2).mean(dim=0) - X.mean(dim=0).pow(2)).mean()
                elif args.diversity_metric == 'difference':
                    loss_diversity = Diversity_loss(logits, target)
                else:
                    raise NotImplementedError
                loss = loss - args.lambda_v * loss_diversity + loss_t
                loss_diversity_total += loss_diversity.item()  # 累积loss_diversity
                if args.loss_dev > 0:
                    loss *= args.loss_dev

                all_logits.append(logits.detach().cpu().numpy())
                all_targets.append(target.detach().cpu().numpy())

            losses.update(loss.item() * args.batch_size, images.size(0))
            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)

        if dist.get_world_size() > 1:
            dist.barrier()

        all_targets = np.concatenate(all_targets)
        all_logits = np.concatenate(all_logits)

        # Apply softmax to logits for saving probabilities
        all_probabilities = F.softmax(torch.tensor(all_logits), dim=2).numpy()

        # Calculate metrics
        metrics, avg_metrics, avg_outputs = calculate_multiclass_metrics(all_targets, all_logits)
        log_and_save_metrics(args=args, metrics=metrics, avg_metrics=avg_metrics, mode='val', logger=logger, epoch=epoch, loss=losses.avg)
        return losses.avg, avg_metrics
def test(train_loader, val_loader, model, args, logger, thresholds):
    
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    # best_threshold = 0.5
    # best_thresholds = [0.5, 0.5]
    saved_data = []
    all_features = []
    all_targets = []
    all_outputs = []  
    net_benefit = []
    all_weights = []  # 用于保存权重
    model_name_xgb = f'/data/wuhuixuan/code/Causal-main/save/XGBoost/xgb_model_fold_{args.fold + 1}.pkl'
    with open(model_name_xgb, 'rb') as model_file:
        xgb_model = pickle.load(model_file)

    model_name_lgbm = f'/data/wuhuixuan/code/Causal-main/save/lgbm/lgbm_model_fold_{args.fold + 1}.pkl'
    with open(model_name_lgbm, 'rb') as model_file:
        lgbm_model = pickle.load(model_file)
    
    # if args.logit_method == args.model_type:
    train_features, train_labels, train_predictions = extract_features(train_loader, model, args)#等会别忘改
    val_features, val_labels, _ = extract_features(val_loader, model, args)
    selected_features_data = pd.read_csv('/data/wuhuixuan/code/Causal-main/data/selected_features_22_with_id_label_fold.csv')
    id_to_index = {str(row['ID']): idx for idx, row in selected_features_data.iterrows()}
    final_outputs = []
    with torch.no_grad():
        for i, (images, clinic_feature, manual_features, target, idx) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            manual_features = manual_features.float().flatten(1).cuda(non_blocking=True)
            clinic_feature = clinic_feature.float().flatten(1).cuda(non_blocking=True)
            with autocast(enabled=args.amp):
                if args.extra_input:
                    extra_input=manual_features
                else:
                    extra_input=None
                if args.use_text:
                    outputs = model(images, extra_input=extra_input, text_input = clinic_feature)
                else:
                    outputs = model(images, extra_input=extra_input)
                output = outputs['output']
                logits = outputs['logits']
                weights = outputs['gate_output']  # 提取权重
            features = outputs['features'].detach().cpu().numpy()
            
            output_cpu = output.detach().cpu().numpy()
            all_features.append(features)
            all_outputs.append(logits.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            # all_weights.append(weights)  # 将权重保存到列表中
            batch_features = []
            mapped_labels = []
            for idx_val in idx.numpy():
                feature_idx = id_to_index.get(str(idx_val) + '.nii.gz')
                if feature_idx is not None:
                    mapped_labels.append(selected_features_data.iloc[feature_idx]['label'])
                    batch_features.append(selected_features_data.drop(columns=['ID', 'label', 'fold']).iloc[feature_idx].values)
                else:
                    mapped_labels.append(None)
                    batch_features.append(np.zeros(selected_features_data.drop(columns=['ID', 'label', 'fold']).shape[1]))

            batch_features = np.array(batch_features)

            xgb_predictions = xgb_model.predict_proba(batch_features)
            lgbm_predictions = lgbm_model.predict_proba(batch_features)

            if args.logit_method == 'XGB':
                final_output = xgb_predictions
            elif args.logit_method == args.model_type:
                final_output = F.softmax(torch.tensor(output_cpu), dim=1).numpy()
            elif args.logit_method == 'LGBM':
                final_output = lgbm_predictions
            elif args.logit_method == 'XGB+' + args.model_type:
                final_output = (args.logit_alpha * F.softmax(torch.tensor(output_cpu), dim=1).numpy() + args.logit_beta * xgb_predictions) / (args.logit_alpha + args.logit_beta)
            elif args.logit_method == 'LGBM+' + args.model_type:
                final_output = (args.logit_alpha * F.softmax(torch.tensor(output_cpu), dim=1).numpy() + args.logit_gamma * lgbm_predictions) / (args.logit_alpha + args.logit_gamma)
            elif args.logit_method == 'XGB+LGBM':
                final_output = (args.logit_beta * xgb_predictions + args.logit_gamma * lgbm_predictions) / (args.logit_beta + args.logit_gamma)
            elif args.logit_method == 'XGB+LGBM+' + args.model_type:
                final_output = (args.logit_alpha * F.softmax(torch.tensor(output_cpu), dim=1).numpy() + args.logit_beta * xgb_predictions + args.logit_gamma * lgbm_predictions) / (args.logit_alpha + args.logit_beta + args.logit_gamma)
            final_outputs.append(final_output)
            # Get softmax outputs for each expert and final output
            expert_outputs = [F.softmax(logits[:, i, :].detach().cpu(), dim=1).numpy()[:, 1] for i in range(logits.shape[1])]
            final_output_class1 = final_output[:, 1]  # Get the probability of class 1 for final output
            weights_list = [weights[:,i].detach().cpu().numpy() for i in range(weights.shape[1])] 
            assert all(len(expert_outputs[i]) == logits.shape[0] for i in range(len(expert_outputs))), "Lengths must match."
            assert all(len(weights_list[i]) == weights.shape[0] for i in range(len(weights_list))), "Lengths must match."

            # 构建 CSV 数据
            for i in range(logits.shape[0]):
                index = idx[i].item()
                target_val = target[i].item()
                row = [index, target_val]
                row.extend(expert_outputs[j][i] for j in range(len(expert_outputs)))
                row.append(final_output_class1[i])
                row.extend(weights_list[j][i] for j in range(len(weights_list)))
                saved_data.append(row)
            # Calculate net benefit for Decision Curve Analysis
            X_test = selected_features_data[selected_features_data['fold'] == f'Fold {args.fold + 1}'].drop(
                columns=['ID', 'label', 'fold'])
            y_test = selected_features_data[selected_features_data['fold'] == f'Fold {args.fold + 1}']['label']
            y_test_prob = xgb_model.predict_proba(X_test)[:, 1]

            net_benefit.append(calculate_net_benefit(y_test, y_test_prob, thresholds))

    # Convert saved_data to DataFrame and save to CSV
    # saved_data = np.array(saved_data)
    save_path = os.path.join(args.output, 'save_data.csv')
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        header = ['ID', 'Target']
        header.extend([f'Expert_{i}' for i in range(len(expert_outputs))])
        header.append('Avg_Output')
        header.extend([f'Weight_{i}' for i in range(len(weights_list))])
        writer.writerow(header)
        
        # 写入数据
        for row in saved_data:
            writer.writerow(row)
    print(save_path)
    all_final_outputs = np.concatenate(final_outputs, axis=0)
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    all_features = np.vstack(all_features)
    all_target_onehot = one_hot(all_targets)
    metrics, avg_metrics, avg_outputs = calculate_multiclass_metrics(all_targets, all_outputs,all_final_outputs)
    if args.display_experts and args.logit_method == args.model_type:
        plot_all_experts_tsne(train_features, train_labels, val_features, val_labels, num_classes=2, experts_range=range(args.num_experts), filename=os.path.join(args.output,'tsne_plot_all_experts.png'))
        softmax_outputs = [F.softmax(torch.tensor(all_outputs[:, expert_idx, :]), dim=1).numpy() for expert_idx in range(args.num_experts)]
        for expert_idx in range(args.num_experts):
            # expert_features = all_features[:, expert_idx, :]#features, labels, num_classes, filename='tsne_plot.png'
            plot_tsne(train_features[: ,expert_idx ,: ], train_labels, val_features[: ,expert_idx ,: ], val_labels, num_classes=2, filename=os.path.join(args.output,f'tsne_plot_expert_{expert_idx}.png'))
            plot_roc_curve(all_target_onehot, softmax_outputs[expert_idx], num_classes=2, filename=os.path.join(args.output,f'roc_curve_expert_{expert_idx}.png'))
            plot_dca_curves(all_targets, softmax_outputs[expert_idx], filename=os.path.join(args.output,f'dca_curves_expert_{expert_idx}.png'))
    else:
        metrics = None
            # avg_output = np.mean(all_outputs, axis=1)
    avg_softmax_output = all_final_outputs
    plot_roc_curve(all_target_onehot, avg_softmax_output, num_classes=2, filename=os.path.join(args.output,'roc_curve_avg_experts.png'))
    plot_pr_curve(all_target_onehot, avg_softmax_output, num_classes=2, filename=os.path.join(args.output,'pr_curve_avg_experts.png'))
    plot_dca_curves(all_targets, avg_softmax_output, filename=os.path.join(args.output,f'dca_curves_expert_avg.png'))
        # plot_tsne(all_features_mean, all_targets, num_classes=2)
    log_and_save_metrics(args=args, metrics=metrics, avg_metrics=avg_metrics, mode='test', logger=logger, epoch=args.epochs,threshold =0.5)

    return metrics, avg_metrics, net_benefit
def main_worker(args, logger):
    """
    This function is the main worker function that takes arguments and a logger as input.
    
    :param args: The `args` parameter typically refers to the command-line arguments passed to a Python
    script when it is executed. These arguments can be accessed using the `sys.argv` list or more
    advanced libraries like `argparse`. The `args` parameter in your function likely contains these
    command-line arguments for further processing
    :param logger: Logger is an object that is used for logging messages, warnings, and errors during
    the execution of the program. It helps in tracking the flow of the program and identifying issues or
    bugs. You can use the logger object to record important events and information while the program is
    running
    """
    global best_mAP
    global best_meanAUC
    global best_f1_score
    # args.resume=f"/data/wuhuixuan/code/Self_Distill_MoE/out/{args.model_type}/{args.criterion_type}/{args.fold+1}/train/{args.logit_method}/model_best.pth.tar"
    # Build model
    if args.model_type=='ResNeXt_Attention_Moe':
        model = ResNeXt_Attention_Moe(num_experts=args.num_experts, nlabels=args.num_class)
    elif args.model_type == 'ResNeXt_Cos_Moe':
        model = ResNeXt_Cos_Moe(num_experts=args.num_experts, nlabels=args.num_class)
    elif args.model_type == 'ResNeXt_Attention_Moe_DCAC':
        model = ResNeXt_Attention_Moe_DCAC(num_experts=args.num_experts, nlabels=args.num_class)
    elif args.model_type == 'ResNeXt_Gating':
        model = ResNeXt_Gating(num_experts=args.num_experts, num_classes=args.num_class, extra_dim = args.extra_dim)
    elif args.model_type == 'ResNeXt_Attention_Gating_Moe':
        model = ResNeXt_Attention_Gating_Moe(num_experts=args.num_experts, nlabels=args.num_class, extra_dim = args.extra_dim)
    elif args.model_type == 'ResNeXt_Attention_Cos_Gating_Moe':
        model = ResNeXt_Attention_Cos_Gating_Moe(num_experts=args.num_experts, nlabels=args.num_class, extra_dim = args.extra_dim)
    elif args.model_type == 'ResNeXt_Attention_Gating_Text_Moe':
        model = ResNeXt_Attention_Gating_Text_Moe(num_experts=args.num_experts, nlabels=args.num_class, extra_dim = args.extra_dim, use_text=args.use_text, text_dim=args.text_dim, text_feature_dim=args.text_feature_dim)
    elif args.model_type == 'ResNeXt50Model':
        model = ResNeXt50Model(num_classes = 2, reduce_dimension = True, use_norm = True, returns_feat = True, num_experts=args.num_experts)
    elif args.model_type == 'ResNeXt_Channel_Attention_Fuse_Gating_Moe':
        model = ResNeXt_Channel_Attention_Fuse_Gating_Moe(num_experts=args.num_experts, nlabels=args.num_class, extra_dim = args.extra_dim, input_size = args.img_size)
    elif args.model_type == 'ResNeXt_Attention_Fuse_Gating_Moe':
        model = ResNeXt_Attention_Fuse_Gating_Moe(num_experts=args.num_experts, nlabels=args.num_class, extra_dim = args.extra_dim, input_size = args.img_size)
    elif args.model_type == 'ResNeXt_Channel_Attention_Gating_Moe':
        model = ResNeXt_Channel_Attention_Gating_Moe(num_experts=args.num_experts, nlabels=args.num_class, extra_dim = args.extra_dim, input_size = args.img_size)
    elif args.model_type=='ResNeXt_Moe':#Efficient_Attention_Gating_Moe
        model = ResNeXt_MoE(num_experts=args.num_experts, num_classes=args.num_class)
    elif args.model_type == 'Efficient_b5_Attention_Gating_Moe':
        model = Efficient_Attention_Gating_Moe(model_name='efficientnet-b5', num_experts=args.num_experts, num_classes=args.num_class, extra_dim=args.extra_dim, pretrained_model_path='/data/wuhuixuan/code/Self_Distill_MoE/pretrain/efficientnet-b2-8bb594d6.pth')
    elif args.model_type == 'Efficient_Attention_Gating_Moe':
        model = Efficient_Attention_Gating_Moe(model_name='efficientnet-b2', num_experts=args.num_experts, num_classes=args.num_class, extra_dim=args.extra_dim, pretrained_model_path = None)#, pretrained_model_path='/data/wuhuixuan/code/Self_Distill_MoE/pretrain/efficientnet-b2-8bb594d6.pth')
    elif args.model_type == 'DenseNet_Attention_Gating_MoE':#resnet50_attention_gating_moe
        model = DenseNet_Attention_Gating_MoE(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), extra_dim = args.extra_dim, num_classes=args.num_class, num_experts = args.num_experts)
    elif args.model_type == 'resnet50_attention_gating_moe':
        model = resnet50_attention_gating_moe(num_classes=args.num_class, include_top=True, use_norm=False, num_experts=args.num_experts, extra_dim = args.extra_dim)
    elif args.model_type == 'densenet201_attention_gating_moe':#resnet50_attention_gating_moe
        model = densenet201_attention_gating_moe(extra_dim = args.extra_dim, num_classes=args.num_class, num_experts = args.num_experts)
    elif args.model_type == 'densenet169_attention_gating_moe':#resnet50_attention_gating_moe
        model = densenet169_attention_gating_moe(extra_dim = args.extra_dim, num_classes=args.num_class, num_experts = args.num_experts)
    elif args.model_type == 'vgg16_attention_gating_moe':
        model = vgg19_attention_gating_moe(num_classes=args.num_class, use_norm=False, extra_dim=args.extra_dim, num_experts = args.num_experts)
    elif args.model_type == 'vgg16_attention_gating_moe':
        model = vgg16_attention_gating_moe(num_classes=args.num_class, use_norm=False, extra_dim=args.extra_dim, num_experts = args.num_experts)
    elif args.model_type == 'vgg19_attention_gating_moe':
        model = vgg19_attention_gating_moe(num_classes=args.num_class, use_norm=False, extra_dim=args.extra_dim, num_experts = args.num_experts)
    if args.use_text:
        if args.model_type not in ['ResNeXt_Attention_Gating_Text_Moe']:
            raise ValueError("The model is not use text.")
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    criterion = build_loss(args.criterion_type, args)
    if args.use_criterion_total:
        criterion_total = build_loss('mixed', args) #加入
    else:
        criterion_total = None
    # Optimizer
    args.lr_mult = args.batch_size / 256#根据batch_size大小调整学习率
    if args.optim == 'AdamW':
        param_dicts = [
            {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam_twd':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
    else:
        raise NotImplementedError

    # Tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None

    if args.resume:
        if os.path.isfile(args.resume):#读取模型参数
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                state_dict = checkpoint
            logger.info("Omitting {}".format(args.resume_omit))
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.module.load_state_dict(state_dict, strict=False)
            
            del checkpoint
            del state_dict
            torch.cuda.empty_cache()
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    best_result = []
    best_meanAUC = 0

    train_dataset = DatasetSliceTextManual(
        image_path=args.image_path,
        fold_json=args.fold_json,
        manual_csv_path=args.manual_csv_path,
        csv_path=args.csv_path,
        fold=args.fold, mode='train'
    )
    val_dataset = DatasetSliceTextManual(
        image_path=args.image_path,
        fold_json=args.fold_json,
        manual_csv_path=args.manual_csv_path,
        csv_path=args.csv_path,
        fold=args.fold, mode='val'
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler, drop_last=True
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, sampler=val_sampler, drop_last=True
    )

    if args.evaluate:
        args.output = os.path.join(args.output, 'val')#测试得到的结果放到这个文件中
        os.makedirs(args.output, exist_ok=True)
        args.output = os.path.join(args.output, args.logit_method)
        os.makedirs(args.output, exist_ok=True)
        metrics, avg_metrics, net_benefit = test(train_loader, val_loader, model, args, logger, thresholds=0.5)
        logger.info(' * Average Metrics: {}'.format(avg_metrics))
        return
    # Criterion难道十折
    args.output = os.path.join(args.output, 'train')
    #  args.output = os.path.join(args.output, args.model_type)
    if not os.path.exists(args.output):  # 加入模型类型
        os.makedirs(args.output)
    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses],
        prefix='=> Test Epoch: '
    )

    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                        epochs=args.epochs, pct_start=0.2)

    end = time.time()
    best_epoch = -1
    best_regular_epoch = -1
    train_loss_list = []
    val_loss_list1 = []
    val_loss_list2 = []
    torch.cuda.empty_cache()

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if args.ema_epoch == epoch:
            ema_m = ModelEma(model.module, args.ema_decay)
            torch.cuda.empty_cache()

        startt = time.time()
        loss, train_metrics = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger,criterion_total = criterion_total)
        train_loss_list.append(loss)
        endt = time.time()
        logger.info("Time used：    {} seconds".format(endt - startt))

        if summary_writer:
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        if epoch % args.val_interval == 0:
            val_loss, val_metrics = validate(val_loader, model, criterion, args, logger, epoch, criterion_total = criterion_total)
            val_loss_list1.append(val_loss)
            val_loss_ema, val_metrics_ema = validate(val_loader, ema_m.module, criterion, args, logger, epoch, criterion_total = criterion_total)
            val_loss_list2.append(val_loss_ema)

            losses.update(val_loss)
            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))
            progress.display(epoch, logger)

            save_path = os.path.join(args.output, args.model_type)
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, 'checkpoint.pth.tar')

            is_best = False
            if val_metrics['AUC'] > val_metrics_ema['AUC']:
                if val_metrics['AUC'] > best_meanAUC:
                    best_meanAUC = val_metrics['AUC']
                    best_epoch = epoch
                    is_best = True
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_meanAUC': best_meanAUC,
                        'optimizer': optimizer.state_dict(),
                    }, is_best=True, filename=filename)
                    logger.info("{} | Set best meanAUC {} in ep {}".format(epoch, best_meanAUC, best_epoch))
            else:
                if val_metrics_ema['AUC'] > best_meanAUC:
                    best_meanAUC = val_metrics_ema['AUC']
                    best_epoch = epoch
                    is_best = True
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': ema_m.module.state_dict(),
                        'best_meanAUC': best_meanAUC,
                        'optimizer': optimizer.state_dict(),
                    }, is_best=True, filename=filename)
                    logger.info("{} | Set best meanAUC {} in ep {}".format(epoch, best_meanAUC, best_epoch))

            if not is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_meanAUC': best_meanAUC,
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=filename)

            if math.isnan(val_loss) :
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_meanAUC': best_meanAUC,
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(args.output, 'checkpoint_nan.pth.tar'))
                logger.info('Loss is NaN, break')
                sys.exit(1)

            if args.early_stop and epoch - best_epoch > 8:
                logger.info("Early stopping at epoch {} with best epoch at {}".format(epoch, best_epoch))
                break

    print("Best mAP:", best_mAP)
    best_result.append(best_mAP)
    plot_losses(train_loss_list, val_loss_list1, val_loss_list2, args.output)
    if summary_writer:
        summary_writer.close()

    best_mAP = sum(best_result) / len(best_result)
    print("Best mAP:", best_mAP)
    return 0
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
def main():
    args = get_args()
    #调试添加参数
    # args.weights = [1, 1, 1]
    args.cls_num_list = [150, 224]
    weights_elements = '_'.join(map(str, args.weights))
    if args.use_criterion_total:
        text_criterion_total = f'Use_criterion_total'
        text_path_criterion_total = f'Use_criterion_total'
    else:
        text_criterion_total = f'No_Use_criterion_total'
    text_focalweight = f"Focal_weight_{args.focal_weight}"
    args.result_file = f'/data/wuhuixuan/code/Self_Distill_MoE/result/MultiExperts_{args.model_type}_{args.num_experts}_{args.criterion_type}_{weights_elements}_{args.diversity_metric}_{text_criterion_total}_{text_focalweight}_Adjust_threshold.csv'
    
        
    # 初始化输出路径
    output_path = args.output

    # 逐步构建输出路径并确保每个子目录存在
    output_path = ensure_directory_exists(os.path.join(output_path, args.model_type))
    output_path = ensure_directory_exists(os.path.join(output_path, f'num_expert{args.num_experts}'))
    output_path = ensure_directory_exists(os.path.join(output_path, args.criterion_type))
    output_path = ensure_directory_exists(os.path.join(output_path, str(args.fold + 1)))
    output_path = ensure_directory_exists(os.path.join(output_path, weights_elements))
    output_path = ensure_directory_exists(os.path.join(output_path, args.diversity_metric))
    output_path = ensure_directory_exists(os.path.join(output_path, text_path_criterion_total))
    output_path = ensure_directory_exists(os.path.join(output_path, text_focalweight))

    # 更新 args.output
    args.output = output_path

    print(args.output)
    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="CAUSAL")
    logger.info("Command: " + ' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
    logger.info('local_rank: {}'.format(args.local_rank))

    return main_worker(args, logger)




if __name__ == '__main__':
    main()
    