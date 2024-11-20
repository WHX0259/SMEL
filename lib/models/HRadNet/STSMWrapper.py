import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np

# 假设你已经有了一个交叉熵损失函数
ce_loss = nn.CrossEntropyLoss()

class STSMWrapper(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 H: int = 64,
                 Q: float = 0.01,
                 weight: float = 1,
                 eps: float = 1e-8):
        super().__init__()
        self.model = model
        self.H = H
        self.Q = Q
        self.weight = weight
        self.eps = eps
        self.STSMLoss = None

    def forward(self, input: torch.Tensor, target: torch.Tensor, *ext_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取模型的原始输出
        outputs = self.model(input, *ext_data)
        y = outputs["output"]
        
        # 生成扰动
        delta_x = torch.rand(self.H, *input.shape, device=input.device) * 2 * self.Q - self.Q
        xs = input.unsqueeze(dim=0) + delta_x
        
        # 计算扰动后的输出差异
        delta_ys = [(self.model(x, *ext_data)["output"] - y) for x in xs]
        delta_y_sqr = [torch.square(dy + self.eps) for dy in delta_ys]
        
        # 计算SSM项
        stsm = torch.mean(torch.stack(delta_y_sqr))  # H x B x D* -> B x D*
        stsm = torch.sqrt(torch.mean(stsm.flatten()))  # flatten (B x D* -> B*prod(D*)), mean, then sqrt
        stsm_loss = self.weight * stsm
        
        # 计算交叉熵损失
        # 注意：对于nn.CrossEntropyLoss，目标应该是类索引（长整型张量），而不是one-hot编码
        ce = ce_loss(y, target)
        
        # 总损失
        total_loss = ce + stsm_loss
        
        self.STSMLoss = stsm_loss
        return y, total_loss