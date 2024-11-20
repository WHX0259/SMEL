# -*- coding: utf-8 -*-
from __future__ import division

""" 
Creates a ResNeXt Model as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""
#开始改造成MOE了
__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from collections import OrderedDict
from torch.nn import Parameter
#初始化参数
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

from collections import OrderedDict
class StridedConv(nn.Module):
    """
    downsampling conv layer
    """

    def __init__(self, in_planes, planes, use_relu=False) -> None:
        super(StridedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)

        if self.use_relu:
            out = self.relu(out)

        return out

class ShallowExpert(nn.Module):
    """
    shallow features alignment wrt. depth
    """

    def __init__(self, input_dim=None, depth=None) -> None:
        super(ShallowExpert, self).__init__()
        self.convs = nn.Sequential(
            OrderedDict([(f'StridedConv{k}', StridedConv(in_planes=input_dim * (2 ** k), planes=input_dim * (2 ** (k + 1)), use_relu=(k != 1))) for
                         k in range(depth)]))#通道数都是原来的2倍吗

    def forward(self, x):
        out = self.convs(x)
        return out


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))#直接相乘
        return out
class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)

def weighted_expert_output(expert_outputs, weights):
    # 将每个专家的输出从计算图中分离出来,最好在后期进行微调
    detached_outputs = [output.detach() for output in expert_outputs]
    
    # 加权操作
    weighted_output = sum(w * o for w, o in zip(weights, detached_outputs))
    
    return weighted_output
#啊哈模型改好了
#加门控，首先对数据进行处理，然后经过模块得到输出，输出权重进行加权
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_experts)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        weights = self.softmax(x)
        return weights

class ResNeXt_Gating(nn.Module):
    def __init__(self, num_experts=None, cardinality=8, depth=29, num_classes=1, base_width=64, widen_factor=4, use_norm=False, extra_dim=None):
        super(ResNeXt_Gating, self).__init__()
        self.s = 1
        self.num_experts = num_experts
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)

        if num_experts:
            self.layer3s = nn.ModuleList([self.block('stage_3', self.stages[2], self.stages[3], 2) for _ in range(self.num_experts)])
            if use_norm:
                self.s = 30
                self.classifiers = nn.ModuleList([NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
                self.rt_classifiers = nn.ModuleList([NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
            else:
                self.classifiers = nn.ModuleList([nn.Linear(self.stages[3], num_classes, bias=True) for _ in range(self.num_experts)])
        else:
            self.layer3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
            self.linear = NormedLinear(self.stages[3], num_classes) if use_norm else nn.Linear(self.stages[3], num_classes, bias=True)

        self.depth = list(reversed([i + 1 for i in range(2)]))
        self.exp_depth = [self.depth[i % len(self.depth)] for i in range(self.num_experts)]
        feat_dim = 256
        self.shallow_exps = nn.ModuleList([ShallowExpert(input_dim=feat_dim * (2 ** (d % len(self.depth))), depth=d) for d in self.exp_depth])

        self.gating = GatingNetwork(extra_dim, num_experts)
        self.shallow_avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.apply(_weights_init)

    def block(self, name, in_channels, out_channels, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality, self.base_width, self.widen_factor))
            else:
                block.add_module(name_, ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width, self.widen_factor))
        return block

    def forward(self, x, extra_input=None, crt=False):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        out1 = self.stage_1.forward(x)
        out2 = self.stage_2.forward(out1)
        shallow_outs = [out1, out2]

        if self.num_experts:
            out3s = [self.layer3s[_](out2) for _ in range(self.num_experts)]
            shallow_expe_outs = [self.shallow_exps[i](shallow_outs[i % len(shallow_outs)]) for i in range(self.num_experts)]
            exp_outs = [out3s[i] * shallow_expe_outs[i] for i in range(self.num_experts)]
            exp_outs = [F.avg_pool2d(output, output.size()[3]).view(output.size(0), -1) for output in exp_outs]

            if crt:
                outs = [self.s * self.rt_classifiers[i](exp_outs[i]) for i in range(self.num_experts)]
            else:
                outs = [self.s * self.classifiers[i](exp_outs[i]) for i in range(self.num_experts)]
        else:
            out3 = self.layer3(out2)
            out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
            outs = self.linear(out)

        weights = self.gating(extra_input)
        final_out = weighted_expert_output(outs, weights)

        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights.detach()
            }
        else:
            return final_out
# 测试代码
def test_model():
    model = ResNeXt_Gating(num_experts=3, extra_dim=10)
    input_data = torch.randn(8, 3, 32, 32)  # 假设输入图像是 32x32 的 CIFAR 图像
    extra_input = torch.randn(8, 10)  # 额外输入的特征维度是 10

    output = model(input_data, extra_input)
    print("Final Output:", output["output"])
    print("Logits Shape:", output["logits"].shape)
    print("Features Shape:", output["features"].shape)
    print("Weights Shape:", output["weights"].shape)



class ResNeXt_MoE(nn.Module):#不需要权重，从特征和损失函数入手
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, num_experts=None, cardinality=8, depth=29, num_classes=1, base_width=64, widen_factor=4, use_norm = False):#之后还得看看cosine classifier
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_MoE, self).__init__()
        #前面都是ResNeXt的
        self.s = 1
        self.num_experts = num_experts
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        if num_experts:
            self.layer3s = nn.ModuleList([self.block('stage_3', self.stages[2], self.stages[3], 2) for _ in range(self.num_experts)])
            if use_norm:#是否是用Norm
                self.s = 30
                self.classifiers = nn.ModuleList(#这个是第三层都是64
                    [NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
                self.rt_classifiers = nn.ModuleList(
                    [NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
            else:
                self.classifiers = nn.ModuleList(
                    [nn.Linear(self.stages[3], num_classes, bias=True) for _ in range(self.num_experts)])
        else:
            self.layer3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
            self.linear = NormedLinear(self.stages[3], num_classes) if use_norm else nn.Linear(
                self.stages[3], num_classes, bias=True)
        # self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)#我应该是拿这个开刀

        #加入特征不同层特征融合和MoE

        # self.classifier = nn.Linear(self.stages[3], num_classes)
        self.apply(_weights_init)
        self.depth = list(
            reversed([i + 1 for i in range(2)]))  # [2, 1]
        self.exp_depth = [self.depth[i % len(self.depth)] for i in range(
            self.num_experts)]  # [2, 1, 2]层数不够专家数就循环
        feat_dim =256
        self.shallow_exps = nn.ModuleList([ShallowExpert(
            input_dim=feat_dim * (2 ** (d % len(self.depth))), depth=d) for d in self.exp_depth])

        self.shallow_avgpool = nn.AdaptiveAvgPool2d((8, 8))
        # for key in self.state_dict():
        #     if key.split('.')[-1] == 'weight':
        #         if 'conv' in key:
        #             init.kaiming_normal(self.state_dict()[key], mode='fan_out')
        #         if 'bn' in key:
        #             self.state_dict()[key][...] = 1
        #     elif key.split('.')[-1] == 'bias':
        #         self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.

        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.

        Returns: a Module consisting of n sequential bottlenecks.

        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x, crt=False):
        x = self.conv_1_3x3.forward(x)#16, 64, 256, 256
        x = F.relu(self.bn_1.forward(x), inplace=True)#torch.Size([16, 64, 256, 256])
        out1 = self.stage_1.forward(x)#torch.Size([16, 256, 128, 128])
        out2 = self.stage_2.forward(out1)#torch.Size([16, 512, 64, 64])
        shallow_outs = [out1, out2]
        if self.num_experts:
            out3s = [self.layer3s[_](out2) for _ in range(self.num_experts)]
            shallow_expe_outs = [self.shallow_exps[i](
                shallow_outs[i % len(shallow_outs)]) for i in range(self.num_experts)]#获得的经过处理的浅层特征

            exp_outs = [out3s[i] * shallow_expe_outs[i]#浅层特征和不同卷积得到的深层特征融合
                        for i in range(self.num_experts)]
            exp_outs = [F.avg_pool2d(output, output.size()[3]).view(
                output.size(0), -1) for output in exp_outs]#得到一维特征
            if crt == True:
                outs = [self.s * self.rt_classifiers[i]#经过分类头得到输出
                        (exp_outs[i]) for i in range(self.num_experts)]
            else:
                outs = [self.s * self.classifiers[i]
                        (exp_outs[i]) for i in range(self.num_experts)]
        else:
            out3 = self.layer3(out2)
            out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
            outs = self.linear(out)
        # x = self.stage_3.forward(x)#torch.Size([16, 1024, 32, 32])#正好也是三层，前两层整合一下测试就平均吧
        # self.feat = torch.stack(self.feat, dim=1)
        # self.feat_before_GAP = torch.stack(self.feat_before_GAP, dim=1)
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1)
            }
        else:
            return final_out
if __name__ == '__main__':
    test_model()
    input = torch.randn(16, 3, 256, 256)#图片大小可以改
    net = ResNeXt_MoE(num_experts = 3)#给定自己设计的模型的参数，如果class model(in_channel,out_channel)，就可以写 model(3,1)#3可以换成自己输入数据的通道数，1换成自己输出数据的通道数
    output = net(input)
    #output [16, 1]
    #logit [16, 3, 1]
    print(output['output'].shape)
    print(output['logits'].shape)
    print(output['features'].shape)


# #我不知道是否可以思考一下这个MC蒙特卡洛dropout
# import torch
# import torch.nn as nn
# from torchvision import models

# class ResNeXtMC(nn.Module):
#     def __init__(self):
#         super(ResNeXtMC, self).__init__()
#         self.resnext = models.resnext50_32x4d(pretrained=True)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc = nn.Linear(1000, 1)  # 假设输出为一个值，用于二分类

#     def forward(self, x):
#         x = self.resnext(x)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x
# import torch
# import torch.nn.functional as F
# from sklearn.metrics import roc_auc_score
# import numpy as np
# import pickle
# import pandas as pd
# import os
# from tqdm import tqdm
# #这个是关键，明天训练和验证的时候试试是不是正确的
# def predict_with_uncertainty(model, inputs, n_iter=10):
#     model.train()  # 确保 dropout 层在推理时也保持激活
#     predictions = []

#     for _ in range(n_iter):
#         outputs = model(inputs)
#         predictions.append(torch.sigmoid(outputs).view(-1).cpu().numpy())

#     predictions = np.array(predictions)
#     prediction_mean = predictions.mean(axis=0)
#     prediction_variance = predictions.var(axis=0)
    
#     return prediction_mean, prediction_variance

# def test(val_loader, model, args, logger, thresholds):
#     # Switch model to evaluate mode
#     model.eval()

#     saved_data = []
#     targets = []
#     dl_outputs = []
#     dl_uncertainties = []  # 新增：保存不确定度
#     net_benefit = []  # Initialize net_benefit

#     # Load XGBoost model for the specified fold
#     model_name_xgb = f'/data/wuhuixuan/code/Causal-main/save/XGBoost/xgb_model_fold_{args.fold + 1}.pkl'
#     with open(model_name_xgb, 'rb') as model_file:
#         xgb_model = pickle.load(model_file)

#     # Load LGBM models for all folds
#     model_name_lgbm = f'/data/wuhuixuan/code/Causal-main/save/lgbm/lgbm_model_fold_{args.fold+1}.pkl'
#     with open(model_name_lgbm, 'rb') as model_file:
#         lgbm_model = pickle.load(model_file)

#     # Load selected features data
#     selected_features_data = pd.read_csv('/data/wuhuixuan/code/Causal-main/data/selected_features_22_with_id_label_fold.csv')

#     # Map image ID to index
#     id_to_index = {str(row['ID']): idx for idx, row in selected_features_data.iterrows()}

#     with torch.no_grad():
#         for i, (images, text, _, target, idx) in enumerate(tqdm(val_loader)):
#             images = images.cuda(non_blocking=True)
#             target = target.cuda(non_blocking=True)
#             text['input_ids'] = text['input_ids'].cuda(non_blocking=True)
#             text['attention_mask'] = text['attention_mask'].cuda(non_blocking=True)

#             # Deep learning model prediction with uncertainty
#             output_mean, output_variance = predict_with_uncertainty(model, images, n_iter=100)
#             output_sm = output_mean
#             uncertainty = output_variance

#             batch_features = []
#             mapped_labels = []
#             for idx_val in idx.numpy():
#                 feature_idx = id_to_index.get(str(idx_val) + '.nii.gz')
#                 if feature_idx is not None:
#                     mapped_labels.append(selected_features_data.iloc[feature_idx]['label'])
#                     batch_features.append(selected_features_data.drop(columns=['ID', 'label', 'fold']).iloc[feature_idx].values)
#                 else:
#                     mapped_labels.append(None)
#                     batch_features.append(np.zeros(selected_features_data.drop(columns=['ID', 'label', 'fold']).shape[1]))

#             batch_features = np.array(batch_features)

#             # XGBoost model prediction
#             xgb_predictions = xgb_model.predict_proba(batch_features)[:, 1]

#             # LGBM model prediction
#             lgbm_predictions = lgbm_model.predict_proba(batch_features)[:, 1]

#             # Combine DL, XGBoost, and LGBM predictions
#             if args.logit_method == 'XGB':
#                 final_output = xgb_predictions
#             elif args.logit_method == 'ResNeXt':
#                 final_output = output_sm
#             elif args.logit_method == 'LGBM':
#                 final_output = lgbm_predictions
#             elif args.logit_method == 'XGB+ResNeXt':
#                 final_output = (args.logit_alpha * output_sm + args.logit_beta * xgb_predictions) / (args.logit_alpha + args.logit_beta)
#             elif args.logit_method == 'LGBM+ResNeXt':
#                 final_output = (args.logit_alpha * output_sm + args.logit_gamma * lgbm_predictions) / (args.logit_alpha + args.logit_gamma)
#             elif args.logit_method == 'XGB+LGBM':
#                 final_output = (args.logit_beta * xgb_predictions + args.logit_gamma * lgbm_predictions) / (args.logit_alpha + args.logit_gamma)
#             elif args.logit_method == 'XGB+LGBM+ResNeXt':
#                 final_output = (args.logit_alpha * output_sm + args.logit_beta * xgb_predictions + args.logit_gamma * lgbm_predictions) / (args.logit_alpha + args.logit_beta + args.logit_gamma)

#             for idx_val, final_out, target_val, uncert in zip(idx.numpy(), final_output, target.cpu().numpy(), uncertainty):
#                 saved_data.append([idx_val, final_out, target_val, uncert])
#             targets.extend(target.cpu().numpy())
#             dl_outputs.extend(final_output)
#             dl_uncertainties.extend(uncertainty)

#             # Calculate net benefit for Decision Curve Analysis
#             X_test = selected_features_data[selected_features_data['fold'] == f'Fold {args.fold + 1}'].drop(
#                 columns=['ID', 'label', 'fold'])
#             y_test = selected_features_data[selected_features_data['fold'] == f'Fold {args.fold + 1}']['label']
#             y_test_prob = xgb_model.predict_proba(X_test)[:, 1]

#             net_benefit.append(calculate_net_benefit(y_test, y_test_prob, thresholds))

#     # Save saved_data to file
#     saved_data = np.array(saved_data)
#     with open(f'/data/wuhuixuan/code/Causal-main/result/{args.logit_method}_saved_data.txt', 'a') as f:
#         np.savetxt(f, saved_data, fmt='%s')

#     # Calculate performance metrics
#     targets = np.array(targets)
#     dl_outputs = np.array(dl_outputs)

#     # Calculate ROC AUC score
#     auc_scores = roc_auc_score(targets, dl_outputs)

#     # Calculate other metrics
#     auc_scores, acc_scores, pre_scores, rec_scores, f1_scores, specificity, ap_score = calculate_metrics(targets, dl_outputs)

#     logger.info(f"AUC: {auc_scores}")
#     logger.info(f"ACC: {acc_scores}")
#     logger.info(f"PRE: {pre_scores}")
#     logger.info(f"REC: {rec_scores}")
#     logger.info(f"F1-Score: {f1_scores}")
#     logger.info(f"Specificity: {specificity}")
#     logger.info(f"Average Precision Score: {ap_score}")

#     args.output = os.path.join(args.output, args.logit_method)
#     if not os.path.exists(args.output):  # if the path does not exist
#         os.makedirs(args.output)

#     # Plot ROC curve
#     plot_roc_curve(targets, dl_outputs, args.logit_method, args.fold + 1, args.output)

#     # Plot Precision-Recall curve
#     plot_pr_curve(targets, dl_outputs, args.logit_method, args.fold + 1, args.output)

#     # Plot Decision Curve
#     plot_dca_curves(targets, dl_outputs, args.logit_method, args.fold + 1, args.output)

#     # Save evaluation metrics to CSV file
#     eval_metrics = {
#         'AUC': auc_scores,
#         'Accuracy': acc_scores,
#         'Precision': pre_scores,
#         'Recall': rec_scores,
#         'F1-Score': f1_scores,
#         'Specificity': specificity,
#         'Average Precision Score': ap_score
#     }
#     df = pd.DataFrame([eval_metrics])

#     # Add fold and method columns
#     df['Fold'] = args.fold + 1
#     df['Method'] = args.logit_method

#     # Check if the file exists, and append to it if it does
#     if os.path.exists(args.excel_file_name):
#         df.to_csv(args.excel_file_name, mode='a', header=False, index=False)
#     else:
#         df.to_csv(args.excel_file_name, index=False)

#     return auc_scores, eval_metrics, net_benefit
