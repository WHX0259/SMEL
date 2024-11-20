from HRadNet import HRadNet
from RBF import RBF
from STSMWrapper import STSMWrapper

import torch
import numpy as np
#这个是一个分类头
# 创建一个 RBF 层实例
in_features = 128
n_centers = 10
out_features = 3
rbf_layer = RBF(in_features, n_centers, out_features)

# 生成一些示例输入数据
batch_size = 4
input_data = torch.randn(batch_size, in_features)

# 通过 RBF 层进行前向传播
output = rbf_layer(input_data)

# 打印输入和输出
print("Input Data:")
print(input_data.shape)
print("Output Data:")
print(output.shape)

# 检查输出形状
assert output.shape == (batch_size, out_features)
print("Output shape is correct.")
#下面应该也是一个分类头