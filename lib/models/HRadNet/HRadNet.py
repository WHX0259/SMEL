import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .RBF import RBF
import os
class HRadBlock(nn.Module):
    def __init__(self,
                 inp: int,
                 outp: int,
                 size: int,
                 gsize: int,
                 dropout_ratio: float,
                 linear_scale: int = 1) -> None:
        super(HRadBlock, self).__init__()
        if size == 1:
            conv = nn.Conv2d(inp, outp, size)
        else:
            conv = nn.Conv2d(inp, outp, 7, 2, 3, groups=inp)
        self.downsample = nn.Sequential(
            conv,
            nn.BatchNorm2d(outp),
            # nn.ELU(),
            nn.SELU()
        )
        self.dropout = nn.Dropout(dropout_ratio)
        # self.gconv = nn.Conv2d(outp, outp, gsize, groups=outp)
        self.gconv = nn.Linear(gsize * gsize, linear_scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.downsample(x)
        # f = self.gconv(x)
        f = self.gconv(x.flatten(2))
        x = self.dropout(x)
        return x, f


class HRadNet(nn.Module):
    def __init__(self,
                 size: int,
                 meta_size: int,
                 num_classes: int,
                 in_channels: int = 3,
                 hidden_dim: int = 64,
                 channels: List[int] = None,
                 layers: List[int] = None,
                 dropout_ratio: float = .0,
                 linear_scale: int = 1,
                 stage: int = 1) -> None:
        super(HRadNet, self).__init__()

        self.size = size
        # self.depth = int(math.log2(self.size)) # depth
        # assert size > 0 and ((size & (size - 1)) == 0), "size should be power of 2"
        self.depth = int(math.log2(size & (-size))) # depth, 2^n
        self.scale = int(size / (size & (-size)))

        self.meta_size = meta_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channels = channels or [2**(i + 2) for i in range(self.depth + 1)]

        assert len(self.channels) == self.depth + 1, f"len(channels) shoud be {self.depth + 1}"

        self.layers = layers if layers is not None else list(range(self.depth + 1))
        self.prune_layers = []

        self.dropout_ratio = dropout_ratio
        self.linear_scale = linear_scale
        self.blocks = self._build_blocks()

        out_features = sum(self.channels[i] for i in self.layers)
        self.fuse = nn.Bilinear(out_features * linear_scale + 1, meta_size + 1, self.hidden_dim, False)
    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, math.sqrt(1 / m.weight.numel()))

    def _build_blocks(self) -> nn.ModuleList:
        blocks = [HRadBlock(self.in_channels, self.channels[0], 1, self.size, self.dropout_ratio, self.linear_scale)]
        blocks.extend(
            HRadBlock(self.channels[i],
                      self.channels[i + 1],
                      2**(self.depth - i - 1) + 1,
                      2**(self.depth - i - 1) * self.scale,
                      self.dropout_ratio,
                      self.linear_scale)
            for i in range(self.depth)
        )
        return nn.ModuleList(blocks)

    def prune(self, layers: List[int]) -> None:
        # for p in self.blocks.parameters():
        #     p.requires_grad = False
        self.prune_layers = layers

    def forward(self, x: torch.Tensor,extra_input=None) -> torch.Tensor:
        if isinstance(x, Tuple):#输入按照tuple
            x, m = x
            meta = [m]
        else:
            meta = [extra_input]
        features = [torch.ones(x.shape[0], 1, device=x.device)]

        for i in range(self.depth + 1):
            x, f = self.blocks[i](x)
            if i in self.layers:
                if i in self.prune_layers:
                    f.zero_()
                features.append(f.flatten(start_dim=1))
        
        features = torch.cat(features, dim=1)

        m = [torch.ones(x.shape[0], 1, device=x.device)]
        m.extend(v.view((x.shape[0], -1)) for v in meta)
        m = torch.cat(m, dim=1)
        x = self.fuse(features, m)

        return x

# 主网络类
class HRad_Net(nn.Module):
    def __init__(self,
                 size: int,
                 meta_size: int,
                 num_classes: int,
                 in_channels: int = 3,
                 hidden_dim: int = 64,
                 channels: List[int] = None,
                 layers: List[int] = None,
                 dropout_ratio: float = .0,
                 linear_scale: int = 1,
                 stage: int = 1,
                 model_path: Optional[str] = None) -> None:
        super(HRad_Net, self).__init__()
        
        # 初始化编码器
        self.encoder = HRadNet(size=size,
                               meta_size=meta_size,
                               num_classes=num_classes,
                               in_channels=in_channels,
                               hidden_dim=hidden_dim,
                               channels=channels,
                               layers=layers,
                               dropout_ratio=dropout_ratio,
                               linear_scale=linear_scale,
                               stage=stage)
        
        # 根据阶段初始化分类头
        if stage == 1:
            self.classifier = ClassificationHead(in_features=hidden_dim, hidden_features=hidden_dim, out_features=num_classes)
        else:
            self.classifier = RBFClassificationHead(in_features=hidden_dim, n_centers=5, hidden_features=hidden_dim, out_features=num_classes)

            # 如果model_path不为None并且stage等于2，则加载预训练的编码器参数
            if model_path is not None and os.path.isfile(model_path):
                print("=> loading encoder checkpoint from '{}'".format(model_path))
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                
                # 清理状态字典并只保留编码器部分
                state_dict = clean_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
                encoder_state_dict = {k: v for k, v in state_dict.items() if 'encoder' in k}
                
                # 加载编码器参数
                self.encoder.load_state_dict(encoder_state_dict, strict=False)
                
                print("Encoder loaded successfully.")
                for param in self.encoder.parameters():
                    param.requires_grad = False
                
                print("Encoder loaded and frozen successfully.")
            else:
                print("=> no checkpoint found at '{}' or model_path is None".format(model_path))

    def forward(self, x: torch.Tensor,extra_input=None) -> torch.Tensor:
        x = self.encoder(x,extra_input)
        out = self.classifier(x)
        return {"output":out,
                "feature":x}

# 辅助函数：清理状态字典（假设这个函数已经定义好了）
def clean_state_dict(state_dict: dict) -> dict:
    # 清理状态字典的逻辑
    # 例如移除不需要的键或者进行一些转换
    return state_dict
class ClassificationHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(ClassificationHead, self).__init__()
        # 第一个线性层后跟SELU激活函数
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 输出层后跟Sigmoid激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # 第一个线性层 + SELU激活
        x = F.selu(self.fc1(x))
        # 第二个线性层 + Sigmoid激活
        x = torch.sigmoid(self.fc2(x))
        return x

class RBFClassificationHead(nn.Module):
    def __init__(self, in_features, n_centers, hidden_features, out_features):
        super(RBFClassificationHead, self).__init__()
        # 批归一化层
        self.bn = nn.BatchNorm1d(in_features)
        # SSM-RBF层
        self.rbf = RBF(in_features, n_centers, hidden_features)
        # 输出层后跟Sigmoid激活函数
        self.fc = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # 批归一化
        x = self.bn(x)
        # SSM-RBF层
        x = self.rbf(x)
        # 线性变换 + Sigmoid激活
        x = torch.sigmoid(self.fc(x))
        return x
def test_hradnet():
    # 参数设置
    size = 128
    meta_size = 2  # Normalized_Size 和 Normalized_Age
    num_classes = 5
    in_channels = 5
    batch_size = 4

    # 创建模型
    # HRadNet(size = 32, meta_size = 2, num_classes = 5, in_channels = 5,stage=2)
    model = HRad_Net(size, meta_size, num_classes, in_channels)

    # 生成随机输入数据
    input_image = torch.randn(batch_size, in_channels, size, size)
    input_meta = torch.randn(batch_size, meta_size)

    # 前向传播
    output = model((input_image, input_meta))

    # 检查输出形状
    print(output['output'].shape)
    # assert output.shape == (batch_size, num_classes), f"Output shape is incorrect: {output.shape}"

    print("Test passed. Output shape is correct.")

# 运行测试
if __name__ == "__main__":
    test_hradnet()