# -*- coding: utf-8 -*-
from __future__ import division

""" 
Creates a ResNeXt Model as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import math

from torch.nn.parameter import Parameter
from collections import OrderedDict

def calculate_similarity_scores(logits, targets):
    """
    计算并返回相似度分数。
    
    参数:
    logits -- 模型输出的 logits，形状为 [batch_size, num_experts, num_classes]
    targets -- 真实标签的索引形式，形状为 [batch_size]

    返回:
    similarity_scores -- 相似度分数，形状为 [batch_size, num_experts]
    """
    batch_size, num_experts, num_classes = logits.shape
    
    # 初始化相似度分数矩阵
    similarity_scores = torch.zeros(batch_size, num_experts, device=logits.device)
    
    # one-hot 编码目标标签
    one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
    
    # 计算每个专家的 softmax 概率分布和交叉熵损失
    for i in range(num_experts):
        # 对每个专家的 logits 应用 softmax
        softmax_probs = F.softmax(logits[:, i, :], dim=1)
        
        # 计算交叉熵损失，这里使用 reduction='none' 来获取每个样本的损失
        loss = F.cross_entropy(softmax_probs, one_hot_targets, reduction='none').view(batch_size)
        
        # 将损失转换为相似度分数，并累加到相似度分数矩阵中
        similarity_scores[:, i] = torch.exp(-loss)
    
    # 归一化相似度分数，使每个样本的专家相似度总和为 1
    similarity_scores = F.softmax(similarity_scores, dim=1)
    
    return similarity_scores
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
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

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


class ResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)#16, 64, 256, 256
        x = F.relu(self.bn_1.forward(x), inplace=True)#torch.Size([16, 64, 256, 256])
        x = self.stage_1.forward(x)#torch.Size([16, 256, 128, 128])
        x = self.stage_2.forward(x)#torch.Size([16, 512, 64, 64])
        x = self.stage_3.forward(x)#torch.Size([16, 1024, 32, 32])
        
        x = F.avg_pool2d(x, x.shape[-1], 1)
        feat = x.view(-1, self.stages[3])
        
        x = self.classifier(feat)
        return {"output":x,
                "feature":feat}
import warnings
#改好了
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        if F_l < F_g:
            warnings.warn("Input dimension F_l is smaller than F_g. This may affect the attention mechanism.")

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=int(F_l/F_g), padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi#这个只是attention得到的加入一个残差链接把
class DADW_Module(nn.Module):
    def __init__(self, radiomic_dim, deeplearn_dim, hidden_dim, num_experts):
        super(DADW_Module, self).__init__()
        self.w_1 = nn.Linear(radiomic_dim, hidden_dim)
        self.w_2 = nn.Linear(deeplearn_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, num_experts)
    def forward(self, radiomic_features, deeplearn_features):
        radiomic_features = self.w_1(radiomic_features)
        deeplearn_features = self.w_2(deeplearn_features)
        fuse_features = radiomic_features * deeplearn_features
        weight = self.w(fuse_features)
        return weight

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DeepwiseAttentionFusion(nn.Module):
    def __init__(self, in_channels_list, target_size, target_channels, reduction=16):
        super(DeepwiseAttentionFusion, self).__init__()
        self.attention_blocks = nn.ModuleList([
            ChannelAttention(in_channels, reduction) for in_channels in in_channels_list
        ])
        self.target_size = target_size
        self.fusion_conv = nn.Conv2d(sum(in_channels_list), target_channels, kernel_size=1)

    def forward(self, features):
        # Apply attention on each feature map
        attended_features = [self.attention_blocks[i](features[i]) for i in range(len(features))]
        
        # Upsample or downsample to target size
        resized_features = [F.interpolate(f, size=self.target_size, mode='bilinear', align_corners=False) for f in attended_features]
        
        # Concatenate the attended features
        fused_features = torch.cat(resized_features, dim=1)
        
        # Apply a 1x1 convolution to fuse the features
        output = self.fusion_conv(fused_features)
        return output

import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import warnings
def generate_attention_sequence(num_experts, num_features):
    sequence = []
    for i in range(num_experts-1):
        sequence.append((num_features - 2 - i % (num_features - 1)) % num_features)
    return sequence


def generate_attention_all_sequence(num_experts, num_features):
    sequence = []
    for i in range(num_experts):
        sequence.append((num_features - 1 - i % (num_features)) % num_features)
    return sequence
class Cos_Classifier(nn.Module):
    """ plain cosine classifier """

    def __init__(self,  num_classes=10, in_dim=640, scale=16, bias=False):
        super(Cos_Classifier, self).__init__()
        self.scale = scale
        self.weight = Parameter(torch.Tensor(num_classes, in_dim))
        self.bias = Parameter(torch.Tensor(num_classes), requires_grad=bias)
        self.init_weights()

    def init_weights(self):
        self.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, **kwargs):
        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)#先进行L2归一化
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        out = torch.mm(ex, self.scale * ew.t()) + self.bias
        return out

class ResNeXt_Attention_Cos_Moe(nn.Module):#重新改了改

    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Attention_Cos_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)

        # Initialize classifiers based on num_experts#num_classes=10, in_dim=640, scale=16
        self.classifiers = nn.ModuleList([Cos_Classifier(nlabels, in_dim= self.stages[-1], scale=16) for _ in range(num_experts)])

        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        # feat_indices = list(range(len(self.stages) - 1, -1, -1))[:num_experts - 1]
        index = generate_attention_sequence(self.num_experts, 3)
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx]+1], self.stages[-1], 128))

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = self.stage_3(feat2)
        feat_list = [feat1, feat2, feat3]
        index = generate_attention_sequence(self.num_experts, 3)
        attentioned_features = []
        
        for idx in range(len(index)):  
            attentioned_features.append(self.attention[idx](feat_list[index[idx]],feat3))#idx=0,
        all_features = [feat3] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
                feat.size(0), -1) for feat in all_features]#得到一维特征
        outs = []
        for idx, feat in enumerate([feat3] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1)
            }
        else:
            return final_out

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))#直接相乘
        return out
class ResNeXt_Attention_Moe(nn.Module):#重新改了改
    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4,use_norm=False):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Attention_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)

        # Initialize classifiers based on num_experts#num_classes=10, in_dim=640, scale=16
        if use_norm:#是否是用Norm
            self.s = 30
            self.classifiers = nn.ModuleList(#这个是第三层都是64
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[3], nlabels, bias=True) for _ in range(self.num_experts)])
        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        # feat_indices = list(range(len(self.stages) - 1, -1, -1))[:num_experts - 1]
        index = generate_attention_sequence(self.num_experts, 3)
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx]+1], self.stages[-1], 128))

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = self.stage_3(feat2)
        feat_list = [feat1, feat2, feat3]
        index = generate_attention_sequence(self.num_experts, 3)
        attentioned_features = []
        
        for idx in range(len(index)):  
            attentioned_features.append(self.attention[idx](feat_list[index[idx]],feat3))#idx=0,
        all_features = [feat3] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
                feat.size(0), -1) for feat in all_features]#得到一维特征
        outs = []
        for idx, feat in enumerate([feat3] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1)
            }
        else:
            return final_out


class ResNeXt_Cos_Moe(nn.Module):#重新改了改

    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Cos_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)

        # Initialize classifiers based on num_experts#num_classes=10, in_dim=640, scale=16
        self.classifiers = nn.ModuleList([Cos_Classifier(nlabels, in_dim= self.stages[-1], scale=16) for _ in range(num_experts)])

        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        # feat_indices = list(range(len(self.stages) - 1, -1, -1))[:num_experts - 1]
        # index = generate_attention_sequence(self.num_experts, 3)
        # for idx in range(len(index)):
        #     self.attention.append(Attention_block(self.stages[index[idx]+1], self.stages[-1], 128))

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = self.stage_3(feat2)
        feat_list = [feat3] * 3
        exp_outs = [F.avg_pool2d(feat, feat.size(3)).view(feat.size(0), -1) for feat in feat_list]
        feat_flatten = F.avg_pool2d(feat3, feat3.size(3)).view(feat3.size(0), -1)
        outs = []
        for idx in range(self.num_experts):
            outs.append(self.classifiers[idx](feat_flatten.view(-1, self.stages[3])))
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1)
            }
        else:
            return final_out

import numpy as np
classes_num = 2
source_number = 3
hyper_k1 = source_number*classes_num
hyper_k2 = source_number*classes_num
#改成动态卷积
#就是我也没有多个domain呀，他有标签我没有，以预测值detach作为target？不监督了，直接用？？
class ResNeXt_Attention_Moe_DCAC(nn.Module):#重新改了改

    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Attention_Moe_DCAC, self).__init__()
        self.cardinality = cardinality
        self.quality = [0] * 3#还没有训练时候模型的质量是0，模型胡乱预测
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        # Initialize classifiers based on num_experts#num_classes=10, in_dim=640, scale=16
        self.classifiers = nn.ModuleList([Cos_Classifier(nlabels, in_dim= self.stages[-1], scale=16) for _ in range(num_experts)])

        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        # feat_indices = list(range(len(self.stages) - 1, -1, -1))[:num_experts - 1]
        index = generate_attention_sequence(self.num_experts, 3)
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx]+1], self.stages[-1], 128))
        self.classifier_fc = nn.Sequential(
            nn.Linear(3 * self.stages[3], source_number)
        )#获得domain的权重
        da_patameters = hyper_k1 * hyper_k1 + hyper_k1
        self.da_controller = nn.Conv2d(source_number, da_patameters, kernel_size=1, stride=1, padding=0)#获得参数值
        parameter_numbers = hyper_k1*hyper_k2 + hyper_k2*hyper_k2 + hyper_k2*classes_num + hyper_k2 + hyper_k2 + classes_num
        self.controller = nn.Conv2d(3 * self.stages[3], parameter_numbers, kernel_size=1, stride=1, padding=0)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
    def update_model_quality(self, metrics, args):
        for i in range(self.num_experts):
            delta_quality = (args.acc_weight * float(metrics[i]['Accuracy'] > 0.5) + args.auc_weight * float(metrics[i]['Average AUC']>0.6) + args.spe_weight * float(metrics[i]['Specificity']>0.4))/(args.acc_weight+args.auc_weight+args.spe_weight)
            self.quality[i] = self.quality[i] * (1 - args.update_rate) + args.update_rate * delta_quality#更新一下模型
            print('The quality of the network is '+str(self.quality)+'/n')
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
    def parse_dynamic_params(self, params, in_channels, out_channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                if self.input_type == '3D':
                    weight_splits[l] = weight_splits[l].reshape(num_insts * in_channels, -1, 1, 1, 1)
                else:
                    weight_splits[l] = weight_splits[l].reshape(num_insts * in_channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * in_channels)
            else:
                # out_channels x in_channels x 1 x 1
                if self.input_type == '3D':
                    weight_splits[l] = weight_splits[l].reshape(num_insts * out_channels, -1, 1, 1, 1)
                else:
                    weight_splits[l] = weight_splits[l].reshape(num_insts * out_channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * out_channels)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        if self.input_type == '3D':
            assert features.dim() == 5
        else:
            assert features.dim() == 4

        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            if self.input_type == '3D':
                x = F.conv3d(
                    x, w, bias=b,#w是filter
                    stride=1, padding=0,
                    groups=num_insts
                )
            else:
                x = F.conv2d(
                    x, w, bias=b,
                    stride=1, padding=0,
                    groups=num_insts
                )
            if i < n_layers - 1:
                x = F.group_norm(x, num_groups=num_insts)
                x = F.leaky_relu(x)
        return x
    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = self.stage_3(feat2)
        feat_list = [feat1, feat2, feat3]
        index = generate_attention_sequence(self.num_experts, 3)
        attentioned_features = []
        
        for idx in range(len(index)):  
            attentioned_features.append(self.attention[idx](feat_list[index[idx]],feat3))#idx=0,
        #目前好改的是exp_outs它是feature进行全局平均池化，接下来就是concat了。
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
                feat.size(0), -1) for feat in attentioned_features]+[F.avg_pool2d(feat3, feat3.size()[3]).view(feat3.size(0), -1)] #得到一维特征
        global_f_c = torch.cat(exp_outs, dim=1)
        classifier_f = torch.flatten(global_f_c, 1)
        classifier_out = self.classifier_fc(classifier_f)#classifier[2，2048]
        self.classifier_out = classifier_out#这个就是输出的权重
        #利用classifier_out生成参数，类似于聚类哦
        outs = []
        for idx, feat in enumerate([feat3] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))
        
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "classifier_out": classifier_out,#这个可以使用inverse_label-predict作为监督信号
                "quality": self.quality
            }
        else:
            return final_out

class MCdropClassifier(nn.Module):
    def __init__(self, in_features, num_classes, 
                 bottleneck_dim=512, dropout_rate=0.5, 
                 dropout_type='Bernoulli'):
        super(MCdropClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type

        self.bottleneck_drop = self._make_dropout(dropout_rate, dropout_type)

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim),
            nn.ReLU(),
            self.bottleneck_drop
        )

        self.prediction_layer = nn.Linear(bottleneck_dim, num_classes)

    def _make_dropout(self, dropout_rate, dropout_type):
        if dropout_type == 'Bernoulli':
            return nn.Dropout(dropout_rate)
        elif dropout_type == 'Gaussian':
            return GaussianDropout(dropout_rate)
        else:
            raise ValueError(f'Dropout type not found')

    def activate_dropout(self):
        self.bottleneck_drop.train()

    def forward(self, x):
        hidden = self.bottleneck_layer(x)
        pred = self.prediction_layer(hidden)
        return pred


class GaussianDropout(nn.Module):
    def __init__(self, drop_rate):
        super(GaussianDropout, self).__init__()
        self.drop_rate = drop_rate
        self.mean = 1.0
        self.std = math.sqrt(drop_rate / (1.0 - drop_rate))

    def forward(self, x):
        if self.training:
            gaussian_noise = torch.randn_like(x, requires_grad=False).to(x.device) * self.std + self.mean
            return x * gaussian_noise
        else:
            return x

def weighted_expert_output(expert_outputs, weights):
    # 将每个专家的输出从计算图中分离出来,最好在后期进行微调
    detached_outputs = [output.detach() for output in expert_outputs]
    
    # 加权操作
    stacked_outputs = torch.stack(detached_outputs, dim=1)

    # 使用 einsum 进行加权求和，得到大小为 (4, 2) 的输出
    weighted_output = torch.einsum('ijk,ij->ik', stacked_outputs, weights)

    return weighted_output
#模型改好了
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
        x = self.fc1(x)
        x = torch.relu(x)
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
        weights = weights.detach().mean(dim=0).tolist()
        
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights
            }
        else:
            return final_out
# class ResNeXt_Gating(nn.Module):
#     def __init__(self, num_experts=None, cardinality=8, depth=29, num_classes=1, base_width=64, widen_factor=4, use_norm=False, extra_dim=None):
#         super(ResNeXt_Gating, self).__init__()
#         self.s = 1
#         self.num_experts = num_experts
#         self.cardinality = cardinality
#         self.depth = depth
#         self.block_depth = (self.depth - 2) // 9
#         self.base_width = base_width
#         self.widen_factor = widen_factor
#         self.num_classes = num_classes
#         self.output_size = 64
#         self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

#         self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
#         self.bn_1 = nn.BatchNorm2d(64)
#         self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
#         self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)

#         if num_experts:
#             self.layer3s = nn.ModuleList([self.block('stage_3', self.stages[2], self.stages[3], 2) for _ in range(self.num_experts)])
#             if use_norm:
#                 self.s = 30
#                 self.classifiers = nn.ModuleList([NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
#                 self.rt_classifiers = nn.ModuleList([NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
#             else:
#                 self.classifiers = nn.ModuleList([nn.Linear(self.stages[3], num_classes, bias=True) for _ in range(self.num_experts)])
#         else:
#             self.layer3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
#             self.linear = NormedLinear(self.stages[3], num_classes) if use_norm else nn.Linear(self.stages[3], num_classes, bias=True)

#         self.depth = list(reversed([i + 1 for i in range(2)]))
#         self.exp_depth = [self.depth[i % len(self.depth)] for i in range(self.num_experts)]
#         feat_dim = 256
#         self.shallow_exps = nn.ModuleList([ShallowExpert(input_dim=feat_dim * (2 ** (d % len(self.depth))), depth=d) for d in self.exp_depth])

#         self.gating = GatingNetwork(extra_dim, num_experts)
#         self.shallow_avgpool = nn.AdaptiveAvgPool2d((8, 8))
#         self.apply(_weights_init)

#     def block(self, name, in_channels, out_channels, pool_stride=2):
#         block = nn.Sequential()
#         for bottleneck in range(self.block_depth):
#             name_ = '%s_bottleneck_%d' % (name, bottleneck)
#             if bottleneck == 0:
#                 block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality, self.base_width, self.widen_factor))
#             else:
#                 block.add_module(name_, ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width, self.widen_factor))
#         return block

#     def forward(self, x, extra_input=None, crt=False):
#         x = self.conv_1_3x3.forward(x)
#         x = F.relu(self.bn_1.forward(x), inplace=True)
#         out1 = self.stage_1.forward(x)
#         out2 = self.stage_2.forward(out1)
#         shallow_outs = [out1, out2]

#         if self.num_experts:
#             out3s = [self.layer3s[_](out2) for _ in range(self.num_experts)]
#             shallow_expe_outs = [self.shallow_exps[i](shallow_outs[i % len(shallow_outs)]) for i in range(self.num_experts)]
#             exp_outs = [out3s[i] * shallow_expe_outs[i] for i in range(self.num_experts)]
#             exp_outs = [F.avg_pool2d(output, output.size()[3]).view(output.size(0), -1) for output in exp_outs]

#             if crt:
#                 outs = [self.s * self.rt_classifiers[i](exp_outs[i]) for i in range(self.num_experts)]
#             else:
#                 outs = [self.s * self.classifiers[i](exp_outs[i]) for i in range(self.num_experts)]
#         else:
#             out3 = self.layer3(out2)
#             out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
#             outs = self.linear(out)

#         weights = self.gating(extra_input)
#         final_out = weighted_expert_output(outs, weights)
#         weights = weights.detach().mean(dim=0)
#         # 将 tensor 转换为列表
#         weights = weights.tolist()
#         if self.num_experts:
#             return {
#                 "output": final_out,
#                 "logits": torch.stack(outs, dim=1),
#                 "features": torch.stack(exp_outs, dim=1),
#                 "weights": weights
#             }
#         else:
#             return final_out
class ResNeXt_Attention_Cos_Gating_Moe(nn.Module):

    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4, extra_dim=None):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Attention_Cos_Gating_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)

        # Initialize classifiers based on num_experts
        self.classifiers = nn.ModuleList([Cos_Classifier(nlabels, in_dim=self.stages[-1], scale=16) for _ in range(num_experts)])

        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        index = generate_attention_sequence(self.num_experts, 3)
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx]+1], self.stages[-1], 128))
        self.gating = GatingNetwork(extra_dim, num_experts)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x, extra_input=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = self.stage_3(feat2)
        feat_list = [feat1, feat2, feat3]
        index = generate_attention_sequence(self.num_experts, 3)
        attentioned_features = []
        
        for idx in range(len(index)):  
            attentioned_features.append(self.attention[idx](feat_list[index[idx]], feat3))
        all_features = [feat3] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
                feat.size(0), -1) for feat in all_features]
        outs = []
        for idx, feat in enumerate([feat3] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))
        

        weights = self.gating(extra_input)
    
        final_out = weighted_expert_output(outs, weights)
        weights = weights.detach().mean(dim=0).tolist()
        
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights
            }
        else:
            return final_out
class ResNeXt_Attention_Gating_Moe(nn.Module):
    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4, use_norm=False, extra_dim=None):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Attention_Gating_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)

        # Initialize classifiers based on num_experts
        if use_norm:
            self.s = 30
            self.classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[3], nlabels, bias=True) for _ in range(self.num_experts)])
        
        self.attention = nn.ModuleList()
        # if num_experts > len(self.stages):
        #     num_experts = len(self.stages)
        index = generate_attention_sequence(self.num_experts, 3)
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx] + 1], self.stages[-1], 128))
        
        self.gating = GatingNetwork(extra_dim, num_experts)
        self.gating_feature = GatingNetwork(self.stages[3], num_experts)#就是设置一下门控机制网络的参数
        
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x, extra_input=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = self.stage_3(feat2)
        feat_list = [feat1, feat2, feat3]
        index = generate_attention_sequence(self.num_experts, 3)
        attentioned_features = []
        
        for idx in range(len(index)):
            attentioned_features.append(self.attention[idx](feat_list[index[idx]], feat3))
        
        all_features = [feat3] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
            feat.size(0), -1) for feat in all_features]
        
        outs = []
        for idx, feat in enumerate([feat3] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))

        
        if extra_input is not None:
            weights = self.gating(extra_input)
        else:
            feature_input = F.avg_pool2d(feat3, feat3.size()[3]).view(feat3.size(0), -1)
            weights = self.gating_feature(feature_input)
        final_out = weighted_expert_output(outs, weights)#outs 5 
        weights_detach = weights.detach().mean(dim=0).tolist()
        
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights_detach,
                "gate_output": weights
            }
        else:
            return final_out
        
class ResNeXt_Attention_Last_Gating_Moe(nn.Module):
    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4, use_norm=False, extra_dim=None):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Attention_Last_Gating_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = nn.ModuleList([self.block('stage_3', self.stages[2], self.stages[3], 2) for _ in range(self.num_experts)])

        # Initialize classifiers based on num_experts
        if use_norm:
            self.s = 30
            self.classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[3], nlabels, bias=True) for _ in range(self.num_experts)])
        
        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        index = generate_attention_sequence(self.num_experts, 3)
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx] + 1], self.stages[-1], 128))
        
        self.gating = GatingNetwork(extra_dim, num_experts)
        self.gating_feature = GatingNetwork(self.stages[3], num_experts)#就是设置一下门控机制网络的参数
        
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x, extra_input=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = []
        for idx in range(self.num_experts):
            feat3.append(self.stage_3[idx](feat2))
        feat_list = [feat1, feat2]
        index = generate_attention_all_sequence(self.num_experts, 2)
        attentioned_features = []
        
        for idx in range(len(index)):
            attentioned_features.append(self.attention[idx](feat_list[index[idx]], feat3[idx]))
        
        all_features = [feat3] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
            feat.size(0), -1) for feat in all_features]
        
        outs = []
        for idx, feat in enumerate([feat3] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))

        
        if extra_input == None:
            weights = self.gating(extra_input)
        else:
            feature_input = F.avg_pool2d(feat3, feat3.size()[3]).view(feat3.size(0), -1)
            weights = self.gating_feature(feature_input)
        final_out = weighted_expert_output(outs, weights)
        weights_detach = weights.detach().mean(dim=0).tolist()
        
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights_detach,
                "gate_output": weights
            }
        else:
            return final_out
        

class ResNeXt_Attention_Gating_Text_Moe(nn.Module):

    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4, use_norm=False, extra_dim=None, text_dim=25, text_feature_dim=128, use_text=False):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Attention_Gating_Text_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.use_text = use_text
        if use_text:
            self.text_linear = nn.Linear(text_dim, text_feature_dim, bias=True)
            if use_norm:
                self.s = 30
                self.classifiers = nn.ModuleList(
                    [NormedLinear(self.stages[3] + text_feature_dim, nlabels) for _ in range(self.num_experts)])
                self.rt_classifiers = nn.ModuleList(
                    [NormedLinear(self.stages[3] + text_feature_dim, nlabels) for _ in range(self.num_experts)])
            else:
                self.classifiers = nn.ModuleList(
                    [nn.Linear(self.stages[3] + text_feature_dim, nlabels, bias=True) for _ in range(self.num_experts)])
        else:
            if use_norm:
                self.s = 30
                self.classifiers = nn.ModuleList(
                    [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
                self.rt_classifiers = nn.ModuleList(
                    [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
            else:
                self.classifiers = nn.ModuleList(
                    [nn.Linear(self.stages[3], nlabels, bias=True) for _ in range(self.num_experts)])
        
        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        index = generate_attention_sequence(self.num_experts, 3)
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx] + 1], self.stages[-1], 128))
        
        self.gating = GatingNetwork(extra_dim, num_experts)
        self.gating_feature = GatingNetwork(self.stages[3], num_experts)#就是设置一下门控机制网络的参数
        
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x, extra_input=None, text_input=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = self.stage_3(feat2)
        feat_list = [feat1, feat2, feat3]
        index = generate_attention_sequence(self.num_experts, 3)
        attentioned_features = []
        
        for idx in range(len(index)):
            attentioned_features.append(self.attention[idx](feat_list[index[idx]], feat3))
        
        all_features = [feat3] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
            feat.size(0), -1) for feat in all_features]
        
        if self.use_text:
            if text_input is None:
                raise ValueError("Text input is required when use_text is True.")
            text_feature = self.text_linear(text_input)
            exp_outs = [torch.cat((out, text_feature), dim=-1) for out in exp_outs]
        
        outs = []
        for idx, feat in enumerate(exp_outs):
            outs.append(self.classifiers[idx](feat))

        if extra_input == None:
            weights = self.gating(extra_input)
        else:
            feature_input = F.avg_pool2d(feat3, feat3.size()[3]).view(feat3.size(0), -1)
            weights = self.gating_feature(feature_input)

        final_out = weighted_expert_output(outs, weights)
        weights = weights.detach().mean(dim=0).tolist()

        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights
            }
        else:
            return final_out
class ResNeXt_Attention_Last_ReparamGating_Moe(nn.Module):
    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4, use_norm=False, extra_dim=None):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Attention_Last_ReparamGating_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = nn.ModuleList([self.block('stage_3', self.stages[2], self.stages[3], 2) for _ in range(self.num_experts)])
        # Initialize classifiers based on num_experts
        if use_norm:
            self.s = 30
            self.classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[3], nlabels, bias=True) for _ in range(self.num_experts)])
        
        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        self.index = generate_attention_all_sequence(self.num_experts, 2)
        for idx in range(len(self.index)):
            self.attention.append(Attention_block(self.stages[self.index[idx] + 1], self.stages[-1], 128))
        
        self.gating = GatingNetwork(extra_dim, num_experts)
        self.gating_feature = GatingNetwork(self.stages[3]*num_experts, num_experts)#就是设置一下门控机制网络的参数
        
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x, extra_input=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = []
        for idx in range(self.num_experts):
            feat3.append(self.stage_3[idx](feat2))
        feat_list = [feat1, feat2]
        # index = generate_attention_all_sequence(self.num_experts, 3)
        attentioned_features = []
        
        for idx in range(len(self.index)):
            attentioned_features.append(self.attention[idx](feat_list[self.index[idx]], feat3[idx]))
        
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
            feat.size(0), -1) for feat in attentioned_features]
        
        outs = []
        for idx, feat in enumerate(attentioned_features):
            if self.training:  # Add noise and Dropout only during training
                # Add noise to the features to increase diversity
                noise = torch.randn_like(feat) * 0.1  # You can adjust the noise scale
                feat = feat + noise
                # Add Dropout before the final classification layer
                feat = F.dropout(feat, p=0.3, training=self.training)
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))

        
        if extra_input == None:
            feature_input = feature_input = [F.avg_pool2d(feat, feat.size(3)).view(feat.size(0), -1) for feat in feat3]
            feature_input = torch.cat(feature_input, dim=-1)
            weights = self.gating_feature(feature_input)
        else:
            weights = self.gating(extra_input)
        
        final_out = weighted_expert_output(outs, weights)
        weights_detach = weights.detach().mean(dim=0).tolist()
        
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights_detach,
                "gate_output": weights
            }
        else:
            return final_out
class ResNeXt_Attention_Last_Gating_Moe(nn.Module):
    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4, use_norm=False, extra_dim=None):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Attention_Last_Gating_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = nn.ModuleList([self.block('stage_3', self.stages[2], self.stages[3], 2) for _ in range(self.num_experts)])
        # Initialize classifiers based on num_experts
        if use_norm:
            self.s = 30
            self.classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[3], nlabels, bias=True) for _ in range(self.num_experts)])
        
        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        self.index = generate_attention_all_sequence(self.num_experts, 2)
        for idx in range(len(self.index)):
            self.attention.append(Attention_block(self.stages[self.index[idx] + 1], self.stages[-1], 128))
        
        self.gating = ReparamGatingNetwork(extra_dim, num_experts)
        self.gating_feature = ReparamGatingNetwork(self.stages[3]*num_experts, num_experts)#就是设置一下门控机制网络的参数
        
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x, extra_input=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = []
        for idx in range(self.num_experts):
            feat3.append(self.stage_3[idx](feat2))
        feat_list = [feat1, feat2]
        # index = generate_attention_all_sequence(self.num_experts, 3)
        attentioned_features = []
        
        for idx in range(len(self.index)):
            attentioned_features.append(self.attention[idx](feat_list[self.index[idx]], feat3[idx]))
        
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
            feat.size(0), -1) for feat in attentioned_features]
        
        outs = []
        for idx, feat in enumerate(attentioned_features):
            if self.training:  # Add noise and Dropout only during training
                # Add noise to the features to increase diversity
                noise = torch.randn_like(feat) * 0.1  # You can adjust the noise scale
                feat = feat + noise
                # Add Dropout before the final classification layer
                feat = F.dropout(feat, p=0.3, training=self.training)
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))

        
        if extra_input == None:
            feature_input = feature_input = [F.avg_pool2d(feat, feat.size(3)).view(feat.size(0), -1) for feat in feat3]
            feature_input = torch.cat(feature_input, dim=-1)
            weights = self.gating_feature(feature_input)
        else:
            weights = self.gating(extra_input)
        
        final_out = weighted_expert_output(outs, weights)
        weights_detach = weights.detach().mean(dim=0).tolist()
        
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights_detach,
                "gate_output": weights
            }
        else:
            return final_out
# 明天可以试试重参数化技巧得到的门控机制模块
class ReparamGatingNetwork(nn.Module):#重参数化模块，要是使用的化需要使用KL散度，把生成的分布与0，1高斯分布进行一下正则化
    def __init__(self, input_dim, num_experts):
        super(ReparamGatingNetwork, self).__init__()
        self.fc_mean = nn.Linear(input_dim, num_experts)
        self.fc_logvar = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        weights = mean + epsilon * std
        weights = F.softmax(weights, dim=-1)  # Apply Softmax to normalize weights
        return weights
class ResNeXt_Channel_Attention_Fuse_Gating_Moe(nn.Module):
    #(num_experts=args.num_experts, nlabels=args.num_class, extra_dim = args.extra_dim)
    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4, use_norm=False, extra_dim=None, input_size = 224):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Channel_Attention_Fuse_Gating_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)

        # Initialize classifiers based on num_experts
        if use_norm:
            self.s = 30
            self.classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[3], nlabels, bias=True) for _ in range(self.num_experts)])
        
        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        index = generate_attention_sequence(self.num_experts, 3)
        for idx in range(len(index)):
            self.attention.append(DeepwiseAttentionFusion([self.stages[index[idx] + 1], self.stages[-1]],target_size=(int(input_size/8), int(input_size/8)), target_channels = self.stages[-1]))
        
        self.gating = DADW_Module(extra_dim, sum(self.stages)-self.stages[0], hidden_dim=128, num_experts=num_experts)
        
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x, extra_input=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = self.stage_3(feat2)
        feat_list = [feat1, feat2, feat3]
        index = generate_attention_sequence(self.num_experts, 3)
        attentioned_features = []
        feat_avg = [F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in feat_list]
        combined_features = torch.cat(feat_avg, dim=1)

        for idx in range(len(index)):
            attentioned_features.append(self.attention[idx]([feat_list[index[idx]], feat3]))
        
        all_features = [feat3] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
            feat.size(0), -1) for feat in all_features]
        
        outs = []
        for idx, feat in enumerate([feat3] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))

        weights = self.gating(extra_input, combined_features)
        weights_softmax = F.softmax(weights,dim=1)
        final_out = weighted_expert_output(outs, weights_softmax)
        weights_detach = weights_softmax.detach()
        
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights_detach,
                "gate_output": weights_softmax
            }
        else:
            return final_out
class ResNeXt_Channel_Attention_Gating_Moe(nn.Module):
    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4, use_norm=False, extra_dim=None, input_size = 224):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Channel_Attention_Gating_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)

        # Initialize classifiers based on num_experts
        if use_norm:
            self.s = 30
            self.classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[3], nlabels, bias=True) for _ in range(self.num_experts)])
        
        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        index = generate_attention_sequence(self.num_experts, 3)
        for idx in range(len(index)):
            self.attention.append(DeepwiseAttentionFusion([self.stages[index[idx] + 1], self.stages[-1]],target_size=(int(input_size/8), int(input_size/8)), target_channels = self.stages[-1]))
        
        self.gating = GatingNetwork(extra_dim, num_experts)
        self.gating_feature = GatingNetwork(self.stages[3], num_experts)#就是设置一下门控机制网络的参数
        
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x, extra_input=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = self.stage_3(feat2)
        feat_list = [feat1, feat2, feat3]
        index = generate_attention_sequence(self.num_experts, 3)
        attentioned_features = []
        
        for idx in range(len(index)):
            attentioned_features.append(self.attention[idx]([feat_list[index[idx]], feat3]))
        
        all_features = [feat3] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
            feat.size(0), -1) for feat in all_features]
        
        outs = []
        for idx, feat in enumerate([feat3] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))

        
        if extra_input==None:
            feature_input = F.avg_pool2d(feat3, feat3.size()[3]).view(feat3.size(0), -1)
            weights = self.gating_feature(feature_input)
        else:
            weights = self.gating(extra_input)
        weights_softmax = F.softmax(weights,dim=1)
        final_out = weighted_expert_output(outs, weights_softmax)
        weights_detach = weights_softmax.detach()
        
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights_detach,
                "gate_output": weights_softmax
            }
        else:
            return final_out
class ResNeXt_Attention_Fuse_Gating_Moe(nn.Module):
    def __init__(self, num_experts=3, cardinality=8, depth=29, nlabels=1, base_width=64, widen_factor=4, use_norm=False, extra_dim=None,input_size=None):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt_Attention_Fuse_Gating_Moe, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]
        self.num_experts = num_experts
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)

        # Initialize classifiers based on num_experts
        if use_norm:
            self.s = 30
            self.classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], nlabels) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[3], nlabels, bias=True) for _ in range(self.num_experts)])
        
        self.attention = nn.ModuleList()
        if num_experts > len(self.stages):
            num_experts = len(self.stages)
        index = generate_attention_sequence(self.num_experts, 3)
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx] + 1], self.stages[-1], 128))
        
        self.gating = DADW_Module(extra_dim, sum(self.stages)-self.stages[0], hidden_dim=128, num_experts=num_experts)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

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

    def forward(self, x, extra_input=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        feat1 = self.stage_1(x)
        feat2 = self.stage_2(feat1)
        feat3 = self.stage_3(feat2)
        feat_list = [feat1, feat2, feat3]
        index = generate_attention_sequence(self.num_experts, 3)
        attentioned_features = []
        feat_avg = [F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in feat_list]
        combined_features = torch.cat(feat_avg, dim=1)

        for idx in range(len(index)):
            attentioned_features.append(self.attention[idx](feat_list[index[idx]], feat3))
        
        all_features = [feat3] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
            feat.size(0), -1) for feat in all_features]
        
        outs = []
        for idx, feat in enumerate([feat3] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))

        
        
        weights = self.gating(extra_input, combined_features)
        weights_softmax = F.softmax(weights,dim=1)
        final_out = weighted_expert_output(outs, weights_softmax)
        weights_detach = weights_softmax.detach()
        
        if self.num_experts:
            return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights_detach,
                "gate_output": weights_softmax
            }
        else:
            return final_out
#下面是测试代码
if __name__ == "__main__":
    # 假设上述模型定义在当前的作用域内
    # from your_module import ResNeXt_Attention_Gating_Text_Moe

    # 初始化模型
    num_experts = 5
    cardinality = 8
    depth = 29
    nlabels = 10  # 假设有10个类别
    base_width = 64
    widen_factor = 4
    use_norm = True
    extra_dim = 22
    text_dim = 25
    text_feature_dim = 128
    use_text = True

    model = ResNeXt_Attention_Gating_Moe(
        num_experts=num_experts,
        cardinality=cardinality,
        depth=depth,
        nlabels=nlabels,
        base_width=base_width,
        widen_factor=widen_factor,
        use_norm=use_norm,
        extra_dim=extra_dim,

    )

    # 设置模型为评估模式
    model.eval()

    # 创建随机的输入数据
    batch_size = 1  # 假设批量大小为8
    image_input = torch.randn(batch_size, 3, 128, 128)  # 图片输入
    extra_input = torch.randn(batch_size, extra_dim)  # 额外信息输入
    text_input = torch.randn(batch_size, text_dim)  # 文本输入

    # 执行模型的前向传播
    with torch.no_grad():  # 测试时不需要计算梯度
        outputs = model(image_input, extra_input)

    # 打印输出结果
    print("Output:", outputs["output"])
    print("Logits:", outputs["logits"])
    print("Features:", outputs["features"])
    print("Gating Weights:", outputs["weights"])
# Define the model class here...

# # Create a function to test model outputs with random input
# def test_model_outputs(model, input_shape=(3, 256, 256), num_experts=3):
#     # Generate random input tensor
#     random_input = torch.randn(1, *input_shape)  # Assuming batch size of 1

#     # Get model outputs
#     outputs = model(random_input)

#     # Print output shapes
#     for idx, out in enumerate(outputs):
#         print(f"Output {idx + 1} shape:", out.shape)

# # Example of testing model outputs
# model = ResNeXt_Attention_Moe(num_experts=3, cardinality=8, depth=29, nlabels=10, base_width=64, widen_factor=4)

# # Set the model to evaluation mode
# model.eval()

# # Test model outputs with random input
# test_model_outputs(model)


# print("num_experts=3:", generate_attention_sequence(3, 3))  # Output: [1, 0]
# print("num_experts=4:", generate_attention_sequence(4, 3))  # Output: [1, 0, 1]
# print("num_experts=5:", generate_attention_sequence(5, 3))  # Output: [1, 0, 1, 0]
# print("num_experts=6:", generate_attention_sequence(6, 3))  # Output: [1, 0, 1, 0, 1]
# # Initialize the tensors
# tensor1 = torch.randn(16, 256, 128, 128)
# tensor2 = torch.randn(16, 512, 64, 64)
# tensor3 = torch.randn(16, 1024, 32, 32)

# # Create an instance of the Attention_block
# attention_block = Attention_block(F_g=256, F_l=512, F_int=128)

# # Test the Attention_block with tensor1 and tensor2
# output = attention_block(tensor1, tensor2)

# print("Output shape:", output.shape)
 
# if __name__ == '__main__':
#     input = torch.randn(16, 3, 256, 256)#图片大小可以改
#     net = ResNeXt()#给定自己设计的模型的参数，如果class model(in_channel,out_channel)，就可以写 model(3,1)#3可以换成自己输入数据的通道数，1换成自己输出数据的通道数
#     output = net(input)
#     print(output.shape)
