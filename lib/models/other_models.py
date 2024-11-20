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
import warnings
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
def generate_attention_sequence(num, stage):
    if stage < 2:
        raise ValueError("Stage must be at least 2.")
    
    # 生成初始序列
    base_sequence = list(range(stage - 2, -1, -1))
    
    # 当需要的数大于阶段数时，重复序列直到满足数量要求
    if num-1 > stage:
        full_sequences, remainder = divmod(num-1, len(base_sequence))
        sequence = (base_sequence * full_sequences) + base_sequence[:remainder]
    else:
        sequence = base_sequence[:num-1]
    
    return sequence

# def generate_attention_all_sequence(num_experts, num_features):
#     sequence = []
#     for i in range(num_experts):
#         sequence.append((num_features - 1 - i % (num_features)) % num_features)
#     return sequence
class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))#直接相乘
        return out
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, scale = None):
        super(Attention_block, self).__init__()

        if F_l < F_g:
            warnings.warn("Input dimension F_l is smaller than F_g. This may affect the attention mechanism.")
        if scale == None:
            scale = int(F_l/F_g)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=scale, padding=0, bias=False),
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


class ResNeXt_Attention_Gating_Moe(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
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

        
        if extra_input == None:
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

def weighted_expert_output(expert_outputs, weights):
    # 将每个专家的输出从计算图中分离出来,最好在后期进行微调
    detached_outputs = [output.detach() for output in expert_outputs]
    
    # 加权操作
    stacked_outputs = torch.stack(detached_outputs, dim=1)

    # 使用 einsum 进行加权求和，得到大小为 (4, 2) 的输出
    weighted_output = torch.einsum('ijk,ij->ik', stacked_outputs, weights)#stack_outputs:1,3,2,1,1,3

    return weighted_output

#加油改一个EfficientNet
"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""
################################################################EfficientNet##############################################################
# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import torch
from torch import nn
from torch.nn import functional as F
from .effientNet_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)


VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        >>> import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        # in_channels = block_args.output_filters  # output of final block
        # out_channels = round_filters(1280, self._global_params)
        # Conv2d = get_same_padding_conv2d(image_size=image_size)
        # self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # # Final linear layer
        # self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        # if self._global_params.include_top:
        #     self._dropout = nn.Dropout(self._global_params.dropout_rate)
        #     self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        # x = self._swish(self._bn1(self._conv_head(x)))
        # endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        # x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        # x = self._avg_pooling(x)
        # if self._global_params.include_top:
        #     x = x.flatten(start_dim=1)
        #     x = self._dropout(x)
        #     x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
# class EfficientNet(nn.Module):
#     """EfficientNet Feature Extractor.
#        This class keeps only the feature extraction part of the EfficientNet model,
#        removing the classification head.
#     """

#     def __init__(self, blocks_args=None, global_params=None):
#         super().__init__()
#         assert isinstance(blocks_args, list), 'blocks_args should be a list'
#         assert len(blocks_args) > 0, 'block args must be greater than 0'
#         self._global_params = global_params
#         self._blocks_args = blocks_args

#         # Batch norm parameters
#         bn_mom = 1 - self._global_params.batch_norm_momentum
#         bn_eps = self._global_params.batch_norm_epsilon

#         # Get stem static or dynamic convolution depending on image size
#         image_size = global_params.image_size
#         Conv2d = get_same_padding_conv2d(image_size=image_size)

#         # Stem
#         in_channels = 3  # RGB channels
#         out_channels = round_filters(32, self._global_params)  # number of output channels
#         self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
#         self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
#         image_size = calculate_output_image_size(image_size, 2)

#         # Build blocks
#         self._blocks = nn.ModuleList([])
#         for block_args in self._blocks_args:

#             # Update block input and output filters based on depth multiplier.
#             block_args = block_args._replace(
#                 input_filters=round_filters(block_args.input_filters, self._global_params),
#                 output_filters=round_filters(block_args.output_filters, self._global_params),
#                 num_repeat=round_repeats(block_args.num_repeat, self._global_params)
#             )

#             # The first block needs to take care of stride and filter size increase.
#             self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
#             image_size = calculate_output_image_size(image_size, block_args.stride)
#             if block_args.num_repeat > 1:  # modify block_args to keep same output size
#                 block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
#             for _ in range(block_args.num_repeat - 1):
#                 self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))

#         # Head (final convolutional layer before classification head)
#         in_channels = block_args.output_filters  # output of final block
#         out_channels = round_filters(1280, self._global_params)
#         Conv2d = get_same_padding_conv2d(image_size=image_size)
#         self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

#         # Swish activation function
#         self._swish = MemoryEfficientSwish()

#     def set_swish(self, memory_efficient=True):
#         """Sets swish function as memory efficient (for training) or standard (for export).

#         Args:
#             memory_efficient (bool): Whether to use memory-efficient version of swish.
#         """
#         self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
#         for block in self._blocks:
#             block.set_swish(memory_efficient)

#     def extract_features(self, inputs):
#         """Extract features without classification head.

#         Args:
#             inputs (tensor): Input tensor.

#         Returns:
#             Output features from the EfficientNet model.
#         """
#         # Stem
#         x = self._swish(self._bn0(self._conv_stem(inputs)))

#         # Blocks
#         for idx, block in enumerate(self._blocks):
#             drop_connect_rate = self._global_params.drop_connect_rate
#             if drop_connect_rate:
#                 drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
#             x = block(x, drop_connect_rate=drop_connect_rate)

#         # Head
#         x = self._swish(self._bn1(self._conv_head(x)))

#         return x

#     def forward(self, inputs):
#         """Forward pass of the feature extractor.

#         Args:
#             inputs (tensor): Input tensor.

#         Returns:
#             Extracted features.
#         """
#         return self.extract_features(inputs)

#     @classmethod
#     def from_name(cls, model_name, in_channels=3, **override_params):
#         """Create an efficientnet feature extractor model according to name.

#         Args:
#             model_name (str): Name for efficientnet.
#             in_channels (int): Input data's channel number.
#             override_params (other key word params):
#                 Params to override model's global_params.

#         Returns:
#             An efficientnet feature extractor model.
#         """
#         cls._check_model_name_is_valid(model_name)
#         blocks_args, global_params = get_model_params(model_name, override_params)
#         model = cls(blocks_args, global_params)
#         model._change_in_channels(in_channels)
#         return model

#     @classmethod
#     def from_pretrained(cls, model_name, weights_path=None, advprop=False,
#                         in_channels=3, **override_params):
#         """Create a pretrained efficientnet feature extractor model.

#         Args:
#             model_name (str): Name for efficientnet.
#             weights_path (None or str): Path to pretrained weights.
#             advprop (bool): Whether to load advprop weights.
#             in_channels (int): Input data's channel number.
#             override_params (other key word params):
#                 Params to override model's global_params.

#         Returns:
#             A pretrained efficientnet feature extractor model.
#         """
#         model = cls.from_name(model_name, in_channels=in_channels, **override_params)
#         load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=False, advprop=advprop)
#         model._change_in_channels(in_channels)
#         return model

#     def _change_in_channels(self, in_channels):
#         """Adjust model's first convolution layer to in_channels, if in_channels not equals 3."""
#         if in_channels != 3:
#             Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
#             out_channels = round_filters(32, self._global_params)
#             self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

class CustomEfficientNet(EfficientNet):
    def forward(self, inputs):
        # Extract features at different reduction levels
        endpoints = self.extract_endpoints(inputs)
        # Extract specific features
        feat1 = endpoints['reduction_1']  # Feature from reduction level 1
        feat2 = endpoints['reduction_2']  # Feature from reduction level 2
        feat3 = endpoints['reduction_3']  # Feature from reduction level 3
        feat4 = endpoints['reduction_4']  # Feature from reduction level 4
        feat5 = endpoints['reduction_5']  # Feature from reduction level 5
        
        return [feat1, feat2, feat3, feat4, feat5]
class MultiExpertModel(nn.Module):
    def __init__(self, num_experts,num_classes,extra_dim=None):
        super().__init__()
        self.num_experts = num_experts
        self.stages = [16, 24, 48, 120, 352]
        self.attention = nn.ModuleList()
        # if num_experts > len(self.stages):
        #     num_experts = len(self.stages)
        index = generate_attention_sequence(self.num_experts, len(self.stages))
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx]], self.stages[-1], 128,scale=2**(len(self.stages)-1-index[idx])))
        self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[-1], num_classes, bias=True) for _ in range(self.num_experts)])  # List of classifier heads
        #加油
        self.gating = GatingNetwork(extra_dim, num_experts)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def forward(self, features,extra_input=None):
        feat_list = features
        feat_last = feat_list[-1]  # 最后一层 1， 320， 7， 7

        # Generate attention sequence
        index = generate_attention_sequence(self.num_experts, len(feat_list))
        attentioned_features = []#3,2
        
        for idx in range(len(index)):
            # Apply attention module to the selected features
            shallow_feat = feat_list[index[idx]]
            fuse_feat = self.attention[idx](shallow_feat, feat_last)#[1, 16, 112, 112],[1, 24, 56, 56],[1,48,28,28],[1, 120, 14, 14],[1, 352, 7, 7]
            attentioned_features.append(fuse_feat)

        # Combine base feature with attention-enhanced features
        all_features = [feat_last] + attentioned_features

        # Global average pooling on all features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(feat.size(0), -1) for feat in all_features]#

        # Pass pooled features through expert classifiers
        outs = []
        for idx, feat in enumerate([feat_last] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(feat.size(0), -1)))
        weights = self.gating(extra_input)
        final_out = weighted_expert_output(outs, weights)#outs 5 
        weights_detach = weights.detach().mean(dim=0).tolist()
        return {
                "output": final_out,
                "logits": torch.stack(outs, dim=1),
                "features": torch.stack(exp_outs, dim=1),
                "weights": weights_detach,
                "gate_output": weights
            }
# efficientnet = CustomEfficientNet.from_pretrained('efficientnet-b0',weights_path='/data/wuhuixuan/code/Self_Distill_MoE/pretrain/efficientnet-b0-355c32eb.pth')
# multi_expert_model = MultiExpertModel(num_experts=3, num_classes=2,extra_dim=22)
# extra_input = torch.randn(1, 22)
# inputs = torch.randn(1, 3, 224, 224)  # Example input
# features = efficientnet(inputs)  # Extract features from EfficientNet
# output = multi_expert_model(features,extra_input)  # Get output from the multi-expert model
# print(output["logits"])#[1, 16, 112, 112],[1,24,56,56],[1,40,28,28],[1,112,14,14],[1,320,7,7]
# 合并EfficientNet和MultiExpertModel
class Efficient_Attention_Gating_Moe(nn.Module):
    def __init__(self, model_name, num_experts, num_classes, extra_dim, pretrained_model_path):
        super().__init__()
        # 加载EfficientNet特征提取器
        if pretrained_model_path == None:
            self.feature_extractor = CustomEfficientNet.from_name(model_name = model_name)
        else:
            self.feature_extractor = CustomEfficientNet.from_pretrained(model_name = model_name, weights_path=pretrained_model_path)
        # 多专家模型
        self.multi_expert_model = MultiExpertModel(num_experts=num_experts, num_classes=num_classes, extra_dim=extra_dim)

    def forward(self, inputs, extra_input):
        # 提取EfficientNet的特征
        features = self.feature_extractor(inputs)
        # 将特征输入到MultiExpertModel中进行处理
        output = self.multi_expert_model(features, extra_input)
        return output

# # 实例化合并模型
# combined_model = Efficient_Attention_Gating_Moe(
#     model_name = 'efficientnet-b2',
#     num_experts=3, 
#     num_classes=2, 
#     extra_dim=22, 
#     pretrained_model_path='/data/wuhuixuan/code/Self_Distill_MoE/pretrain/efficientnet-b2-8bb594d6.pth'
# )

# # 测试模型
# extra_input = torch.randn(1, 22)
# inputs = torch.randn(1, 3, 224, 224)  # Example input
# output = combined_model(inputs, extra_input)  # Get output from the combined model

# # # 输出logits
# print(output["logits"])

#################################################DenseNet#################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.parameter import Parameter
import warnings
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):# 第一个参数是输入的通道数，第二个是增长率是一个重要的超参数，它控制了每个密集块中特征图的维度增加量，
        #                第四个参数是Dropout正则化上边的概率
        super(_DenseLayer, self).__init__()# 调用父类的构造方法，这句话的意思是在调用nn.Sequential的构造方法
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),  # 批量归一化
        self.add_module('relu1', nn.ReLU(inplace=True)),     # ReLU层
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),    # 表示其输出为4*k   其中bn_size等于4，growth_rate为k     不改变大小，只改变通道的个数
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),  # 批量归一化
        self.add_module('relu2', nn.ReLU(inplace=True)),         # 激活函数
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),    # 输出为growth_rate：表示输出通道数为k  提取特征
        self.drop_rate = drop_rate
 
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)  # 通道维度连接
 
 
class _DenseBlock(nn.Sequential):  # 构建稠密块
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate): # 密集块中密集层的数量，第二参数是输入通道数量
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
 

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))#直接相乘
        return out
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):# 输入通道数 输出通道数
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
 
def generate_attention_sequence(num, stage):
    if stage < 2:
        raise ValueError("Stage must be at least 2.")
    
    # 生成初始序列
    base_sequence = list(range(stage - 2, -1, -1))
    
    # 当需要的数大于阶段数时，重复序列直到满足数量要求
    if num-1 > stage:
        full_sequences, remainder = divmod(num-1, len(base_sequence))
        sequence = (base_sequence * full_sequences) + base_sequence[:remainder]
    else:
        sequence = base_sequence[:num-1]
    
    return sequence

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, scale = None):
        super(Attention_block, self).__init__()

        if F_l < F_g:
            warnings.warn("Input dimension F_l is smaller than F_g. This may affect the attention mechanism.")
        if scale == None:
            scale = int(F_l/F_g)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=scale, padding=0, bias=False),
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
def weighted_expert_output(expert_outputs, weights):
    # 将每个专家的输出从计算图中分离出来,最好在后期进行微调
    detached_outputs = [output.detach() for output in expert_outputs]
    
    # 加权操作
    stacked_outputs = torch.stack(detached_outputs, dim=1)

    # 使用 einsum 进行加权求和，得到大小为 (4, 2) 的输出
    weighted_output = torch.einsum('ijk,ij->ik', stacked_outputs, weights)

    return weighted_output
# DenseNet网络模型基本结构
class DenseNet_Attention_Gating_MoE(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, use_norm=False, num_classes=4,extra_dim =None, num_experts = 3):
 
        super(DenseNet_Attention_Gating_MoE, self).__init__()
        self.num_experts = num_experts
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
 
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
 
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.stages = [128, 256, 640, 1664]#[1, 128, 28, 28],[1, 256, 14, 14],[1, 640, 7, 7],[1, 1664, 7, 7]
        # Linear layer
        if use_norm:
            self.s = 30
            self.classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[3], num_classes, bias=True) for _ in range(self.num_experts)])
        self.attention = nn.ModuleList()
        # if num_experts > len(self.stages):
        #     num_experts = len(self.stages)
        index = generate_attention_sequence(self.num_experts, len(self.stages))#2,1 
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx]], self.stages[-1], 128,scale=2**(len(self.stages)-2-index[idx])))
        
        self.gating = GatingNetwork(extra_dim, num_experts)
        self.gating_feature = GatingNetwork(self.stages[3], num_experts)#就是设置一下门控机制网络的参数
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
 
    def forward(self, x, extra_input=None):
        features = []
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        # features.append(x)  # 添加初始卷积层后的特征

        # Each denseblock
        for name, module in self.features._modules.items():
            if 'denseblock' in name:
                x = module(x)
                # 添加每个密集块的输出特征#[1, 256, 56, 56],[1, 512, 28, 28],[1, 1024, 14, 14],[1, 1024, 7, 7],[1024, 7,7]
            if 'transition' in name:
                x = module(x)
                features.append(x)
        features.append(x)
        feat_last = features[-1]
        index = generate_attention_sequence(self.num_experts, len(self.stages))
        attentioned_features = []
        
        for idx in range(len(index)):  
            shallow_feat = features[index[idx]]#1, 512, 7, 7
            fuse_feat = self.attention[idx](shallow_feat,feat_last)
            attentioned_features.append(fuse_feat)#idx=0,
        all_features = [feat_last] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
                feat.size(0), -1) for feat in all_features]#得到一维特征
        outs = []
        for idx, feat in enumerate([feat_last] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))
        if extra_input is not None:
            weights = self.gating(extra_input)
        else:
            feature_input = F.avg_pool2d(feat_last, feat_last.size()[3]).view(feat_last.size(0), -1)
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
        # x = self.features.norm5(x)
        # out = F.relu(x, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(x.size(0), -1)
        # out = self.classifier(out)
        
        # return features, #out
 
 
def densenet121_attention_gating_moe(use_norm=False, num_classes=4,extra_dim =None, num_experts = 3):
    model = DenseNet_Attention_Gating_MoE(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), extra_dim = extra_dim, num_classes=num_classes, num_experts = num_experts)
    return model
 
 
def densenet169_attention_gating_moe(use_norm=False, num_classes=4,extra_dim =None, num_experts = 3):
    model = DenseNet_Attention_Gating_MoE(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), extra_dim = extra_dim, num_classes=num_classes, num_experts = num_experts)
    return model
 
 
def densenet201_attention_gating_moe(use_norm=False, num_classes=4,extra_dim =None, num_experts = 3):
    model = DenseNet_Attention_Gating_MoE(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), extra_dim = extra_dim, num_classes=num_classes, num_experts = num_experts)
    return model
 
 
def densenet161_attention_gating_moe(use_norm=False, num_classes=4,extra_dim =None, num_experts = 3):
    model = DenseNet_Attention_Gating_MoE(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), extra_dim = extra_dim, num_classes=num_classes, num_experts = num_experts)
    return model
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # 创建DenseNet模型实例
    model = DenseNet_Attention_Gating_MoE(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), extra_dim = 22, num_classes=2, num_experts = 3)

    # 1.31E-10
    #打印模型结构
    print(model)

    # 计算模型的参数数量
    num_params = count_parameters(model)
    print(f'Number of parameters: {num_params}')

    # 创建一个随机输入张量，假设输入图像大小为 224x224
    input_tensor = torch.randn(1, 3, 224, 224)  # batch size = 1
    extra_input = torch.randn(1, 22)
    # 将输入传递给模型
    output = model(input_tensor, extra_input)

    # 打印输出张量的形状
    print(output['logits'])

    # 检查输出是否符合预期
    # assert output.shape == (1, 2), "Output shape does not match expected shape."
 
######################################################ResNet#########################################################
import torch.nn as nn
import torch
from torch.nn import init
import warnings
import torch.nn.functional as F

from torch.nn.parameter import Parameter
def generate_attention_sequence(num, stage):
    if stage < 2:
        raise ValueError("Stage must be at least 2.")
    
    # 生成初始序列
    base_sequence = list(range(stage - 2, -1, -1))
    
    # 当需要的数大于阶段数时，重复序列直到满足数量要求
    if num-1 > stage:
        full_sequences, remainder = divmod(num-1, len(base_sequence))
        sequence = (base_sequence * full_sequences) + base_sequence[:remainder]
    else:
        sequence = base_sequence[:num-1]
    
    return sequence

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, scale = None):
        super(Attention_block, self).__init__()

        if F_l < F_g:
            warnings.warn("Input dimension F_l is smaller than F_g. This may affect the attention mechanism.")
        if scale == None:
            scale = int(F_l/F_g)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=scale, padding=0, bias=False),
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
def weighted_expert_output(expert_outputs, weights):
    # 将每个专家的输出从计算图中分离出来,最好在后期进行微调
    detached_outputs = [output.detach() for output in expert_outputs]
    
    # 加权操作
    stacked_outputs = torch.stack(detached_outputs, dim=1)

    # 使用 einsum 进行加权求和，得到大小为 (4, 2) 的输出
    weighted_output = torch.einsum('ijk,ij->ik', stacked_outputs, weights)

    return weighted_output

# Resnet 18/34使用此残差块
class BasicBlock(nn.Module):  # 卷积2层，F(X)和X的维度相等
    # expansion是F(X)相对X维度拓展的倍数
    expansion = 1  # 残差映射F(X)的维度有没有发生变化，1表示没有变化，downsample=None

    # in_channel输入特征矩阵的深度(图像通道数，如输入层有RGB三个分量，使得输入特征矩阵的深度是3)，out_channel输出特征矩阵的深度(卷积核个数)，stride卷积步长，downsample是用来将残差数据和卷积数据的shape变的相同，可以直接进行相加操作。
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN层在conv和relu层之间

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out=F(X)+X
        out += identity
        out = self.relu(out)

        return out


# Resnet 50/101/152使用此残差块
class Bottleneck(nn.Module):  # 卷积3层，F(X)和X的维度不等
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    """
    # expansion是F(X)相对X维度拓展的倍数
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        # 此处width=out_channel

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # downsample是用来将残差数据和卷积数据的shape变的相同，可以直接进行相加操作。
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out=F(X)+X
        out += identity
        out = self.relu(out)

        return out

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))#直接相乘
        return out
class ResNet(nn.Module):

    def __init__(self,
                 block,  # 使用的残差块类型
                 blocks_num,  # 每个卷积层，使用残差块的个数
                 num_classes=1000,  # 训练集标签的分类个数
                 include_top=True,  # 是否在残差结构后接上pooling、fc、softmax
                 groups=1,
                 width_per_group=64,
                 use_norm = False,
                 num_experts = 3,
                 extra_dim = None
                 ):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 第一层卷积输出特征矩阵的深度，也是后面层输入特征矩阵的深度
        self.num_experts = num_experts
        self.groups = groups
        self.width_per_group = width_per_group

        # 输入层有RGB三个分量，使得输入特征矩阵的深度是3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # _make_layer(残差块类型，残差块中第一个卷积层的卷积核个数，残差块个数，残差块中卷积步长)函数：生成多个连续的残差块的残差结构
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        #这个只是针对resnet50的
        self.stages = [256, 512, 1024, 2048]
        # Initialize classifiers based on num_experts
        if use_norm:
            self.s = 30
            self.classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[3], num_classes, bias=True) for _ in range(self.num_experts)])
        
        self.attention = nn.ModuleList()
        # if num_experts > len(self.stages):
        #     num_experts = len(self.stages)
        index = generate_attention_sequence(self.num_experts, 3)
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx]], self.stages[-1], 128))
        
        self.gating = GatingNetwork(extra_dim, num_experts)
        # self.gating_feature = GatingNetwork(self.stages[3], num_experts)#就是设置一下门控机制网络的参数
        
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        # if self.include_top:  # 默认为True，接上pooling、fc、softmax
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化下采样，无论输入矩阵的shape为多少，output size均为的高宽均为1x1
        #     # 使矩阵展平为向量，如（W,H,C）->(1,1,W*H*C)，深度为W*H*C
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层，512 * block.expansion为输入深度，num_classes为分类类别个数

        # for m in self.modules():  # 初始化
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # _make_layer()函数：生成多个连续的残差块，(残差块类型，残差块中第一个卷积层的卷积核个数，残差块个数，残差块中卷积步长)
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        # 寻找：卷积步长不为1或深度扩张有变化，导致F(X)与X的shape不同的残差块，就要对X定义下采样函数，使之shape相同
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        # layers用于顺序储存各连续残差块
        # 每个残差结构，第一个残差块均为需要对X下采样的残差块，后面的残差块不需要对X下采样
        layers = []
        # 添加第一个残差块，第一个残差块均为需要对X下采样的残差块
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion
        # 后面的残差块不需要对X下采样
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # 以非关键字参数形式，将layers列表，传入Sequential(),使其中残差块串联为一个残差结构
        return nn.Sequential(*layers)

    def forward(self, x, extra_input=None):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)
        features = []
        x = self.layer1(x)#1,256,56,56
        features.append(x)
        x = self.layer2(x)#1, 512, 28, 28
        features.append(x)
        x = self.layer3(x)#1, 1024, 14, 14
        features.append(x)
        x = self.layer4(x)#1, 2048, 7, 7
        features.append(x)
        index = generate_attention_sequence(self.num_experts, 3)
        attentioned_features = []
        feat3_last = features[-1]
        for idx in range(len(index)):
            attentioned_features.append(self.attention[idx](features[index[idx]], feat3_last))
        
        all_features = [feat3_last] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
            feat.size(0), -1) for feat in all_features]
        
        outs = []
        for idx, feat in enumerate([feat3_last] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))

        
        weights = self.gating(extra_input)
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
        # if self.include_top:  # 一般为True
        #     x = self.avgpool(x)
        #     x = torch.flatten(x, 1)
        #     x = self.fc(x)

        # return x

# 至此resnet的基本框架就写好了
# ——————————————————————————————————————————————————————————————————————————————————
# 下面定义不同层的resnet


def resnet50_attention_gating_moe(num_classes=1000, include_top=True,use_norm = False,num_experts = 3,extra_dim = None):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top,use_norm=use_norm,extra_dim=extra_dim,num_experts=num_experts)


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

# def test_resnet50():
#     # 创建resnet50模型实例
#     model = resnet50_attention_gating_moe(num_classes=2, include_top=True,use_norm = False,
#                  num_experts = 3,
#                  extra_dim = 22)

#     # 随机生成一个大小为 (batch_size, channels, height, width) 的张量作为输入
#     batch_size = 1
#     input_tensor = torch.randn(batch_size, 3, 224, 224)
#     extra_input = torch.randn(batch_size, 22)
#     # 将输入张量传递给模型
#     output = model(input_tensor, extra_input)

#     # 输出模型预测的形状
#     print("Output shape:", output['output'].shape)

# # 运行测试
# test_resnet50()
########################################VGG16#######################################################
# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import cast, Dict, List, Union

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "VGG",
    "vgg11", "vgg13", "vgg16", "vgg19",
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
]

vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
from torch.nn import init
import warnings
import torch.nn.functional as F

from torch.nn.parameter import Parameter
def generate_attention_sequence(num, stage):
    if stage < 2:
        raise ValueError("Stage must be at least 2.")
    
    # 生成初始序列
    base_sequence = list(range(stage - 2, -1, -1))
    
    # 当需要的数大于阶段数时，重复序列直到满足数量要求
    if num-1 > stage:
        full_sequences, remainder = divmod(num-1, len(base_sequence))
        sequence = (base_sequence * full_sequences) + base_sequence[:remainder]
    else:
        sequence = base_sequence[:num-1]
    
    return sequence

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, scale = None):
        super(Attention_block, self).__init__()

        if F_l < F_g:
            warnings.warn("Input dimension F_l is smaller than F_g. This may affect the attention mechanism.")
        if scale == None:
            scale = int(F_l/F_g)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=scale, padding=0, bias=False),
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
def weighted_expert_output(expert_outputs, weights):
    # 将每个专家的输出从计算图中分离出来,最好在后期进行微调
    detached_outputs = [output.detach() for output in expert_outputs]
    
    # 加权操作
    stacked_outputs = torch.stack(detached_outputs, dim=1)

    # 使用 einsum 进行加权求和，得到大小为 (4, 2) 的输出
    weighted_output = torch.einsum('ijk,ij->ik', stacked_outputs, weights)

    return weighted_output


def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in vgg_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v

    return layers
class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))#直接相乘
        return out

class VGG(nn.Module):
    def __init__(self, vgg_cfg: List[Union[str, int]], batch_norm: bool = False, num_classes: int = 1000, use_norm=False, 
                 extra_dim=22, num_experts = 3) -> None:
        super(VGG, self).__init__()
        self.features = _make_layers(vgg_cfg, batch_norm)
        self.stages = [64, 128, 256, 512, 512]
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.num_experts = num_experts
        # Initialize classifiers based on num_experts
        if use_norm:
            self.s = 30
            self.classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                [NormedLinear(self.stages[3], num_classes) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.stages[3], num_classes, bias=True) for _ in range(self.num_experts)])
        
        self.attention = nn.ModuleList()
        # if num_experts > len(self.stages):
        #     num_experts = len(self.stages)
        index = generate_attention_sequence(self.num_experts, 5)
        for idx in range(len(index)):
            self.attention.append(Attention_block(self.stages[index[idx]], self.stages[-1], 128,scale =2**(len(self.stages)-1-index[idx]) ))
        
        self.gating = GatingNetwork(extra_dim, num_experts)
        self.gating_feature = GatingNetwork(self.stages[3], num_classes)#就是设置一下门控机制网络的参数
        
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def forward(self, x: Tensor, extra_input=None) -> Tensor:
        features = self._forward_impl(x)
        index = generate_attention_sequence(self.num_experts, 5)
        attentioned_features = []
        feat_last = features[-1]
        for idx in range(len(index)):
            shallow_feat = features[index[idx]]
            fuse_feat = self.attention[idx](shallow_feat, feat_last)
            attentioned_features.append(fuse_feat)
        
        all_features = [feat_last] + attentioned_features
        exp_outs = [F.avg_pool2d(feat, feat.size()[3]).view(
            feat.size(0), -1) for feat in all_features]
        
        outs = []
        for idx, feat in enumerate([feat_last] + attentioned_features):
            feat = F.avg_pool2d(feat, feat.shape[-1], 1)
            outs.append(self.classifiers[idx](feat.view(-1, self.stages[3])))

        
        weights = self.gating(extra_input)
        
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

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        features = []  # 用来保存每一层的特征
        for module in self.features:
            x = module(x)
            if isinstance(module, nn.MaxPool2d):
                features.append(x)#[1, 64, 112,112],[1, 128, 56, 56],[1, 256, 28, 28],[1, 512, 14, 14],[1, 512, 7, 7]
        # out = self.avgpool(x)
        # out = torch.flatten(out, 1)
        # out = self.classifier(out)
        return features  # 返回最终输出和所有特征层的特征

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


def vgg11(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg11"], False, **kwargs)

    return model


def vgg13(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg13"], False, **kwargs)

    return model


def vgg16_attention_gating_moe(num_classes=2, use_norm=False, extra_dim=22, num_experts = 3) -> VGG:
    model = VGG(vgg_cfgs["vgg16"], False, num_classes=num_classes, use_norm=use_norm ,
                 extra_dim=extra_dim, num_experts = num_experts)

    return model


def vgg19_attention_gating_moe(num_classes=2, use_norm=False, extra_dim=22, num_experts = 3) -> VGG:
    model = VGG(vgg_cfgs["vgg19"], False, num_classes=num_classes, use_norm=use_norm ,
                 extra_dim=extra_dim, num_experts = num_experts)

    return model


def vgg11_bn(num_classes=2, use_norm=False, extra_dim=22, num_experts = 3) -> VGG:
    model = VGG(vgg_cfgs["vgg11"], False, num_classes=num_classes, use_norm=use_norm ,
                 extra_dim=extra_dim, num_experts = num_experts)

    return model


def vgg13_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg13"], True, **kwargs)

    return model


def vgg16_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg16"], True, **kwargs)

    return model


def vgg19_bn(num_classes=2, use_norm=False, extra_dim=22, num_experts = 3) -> VGG:
    model = VGG(vgg_cfgs["vgg19"], True,num_classes=num_classes, use_norm=use_norm ,
                 extra_dim=extra_dim, num_experts = num_experts)

    return model
def test_vgg16():
    # 创建 VGG16 模型实例
    model = vgg16_attention_gating_moe(num_classes=2)

    # 随机生成一个大小为 (batch_size, channels, height, width) 的张量作为输入
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    extra_tensor = torch.randn(batch_size, 22)
    # 将输入张量传递给模型
    output = model(input_tensor, extra_tensor)

    # 输出模型预测的形状
    print("Output shape:", output['logits'])

# 运行测试
test_vgg16()