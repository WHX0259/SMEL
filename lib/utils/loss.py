import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
eps = 1e-7
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return torch.mean(entropy)
def diversity_loss(y_pred):
        # Calculate binary cross entropy loss
        y_pred_1 = y_pred
        y_pred_0 = 1 - y_pred
        entropy = Entropy(y_pred_1)+Entropy(y_pred_0)
        return entropy
def ce_loss(output, label):
    if isinstance(output, list):
        total_loss = 0.0
        for out in output:
            total_loss += F.cross_entropy(out, label)
        return total_loss / len(output)
    else:
        return F.cross_entropy(output, label)

def custom_kl_div(prediction, target):
    output_pos = target * (target.clamp(min=1e-7).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=0)
    return output.mean()

def label_smoothing(inputs, epsilon=0.1):
    """
    Applies label smoothing to the input labels.
    
    Args:
        inputs (Tensor): Original labels, shape (N, C) where N is the batch size and C is the number of classes.
        epsilon (float): Smoothing factor.
        
    Returns:
        Tensor: Smoothed labels, same shape as inputs.
    """
    K = inputs.size(-1)  # Number of classes
    return ((1 - epsilon) * inputs) + (epsilon / K)

class MixedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, focal_weight=1.0):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.focal_weight = focal_weight
    def forward(self, y_pred, y_true, weight = None):
        """
        Forward pass for mixed Cross-Entropy Loss and Focal Loss for multi-class classification
        :param y_pred: Predicted logits, shape [batch_size, num_classes] or list of tensors
        :param y_true: True labels, shape [batch_size] or list of tensors
        :return: Mixed Loss
        """
        if isinstance(y_pred, list):
            if weight==None:
                weight = [1]*len(y_pred)
            else:
                if len(y_pred) != len(weight):
                    if len(y_pred) != weight.shape[1]:
                        raise ValueError("Number of logits and weights must match")
                    else:
                        weight = weight.mean(dim=0)
                        # 将 tensor 转换为列表
                        weight = weight.tolist()
            total_loss = 0.0
            for i in range(len(y_pred)):
                ce_loss = F.cross_entropy(y_pred[i], y_true, weight=self.weight, reduction='none')
                
                # Calculate probabilities
                prob = torch.softmax(y_pred[i], dim=1)
                # Get the probabilities of the true classes
                prob_true = prob.gather(1, y_true.unsqueeze(1)).squeeze(1)
                
                focal_weights = self.alpha * (1 - prob_true)**self.gamma
                focal_loss = focal_weights * ce_loss
                
                # Combine CE loss and Focal loss
                mixed_loss = ce_loss + self.focal_weight*focal_loss
                total_loss += weight[i]*mixed_loss.mean()
            
            return total_loss / len(y_pred)
        else:
            ce_loss = F.cross_entropy(y_pred, y_true, weight=self.weight, reduction='none')
            
            # Calculate probabilities
            prob = torch.softmax(y_pred, dim=1)
            # Get the probabilities of the true classes
            prob_true = prob.gather(1, y_true.unsqueeze(1)).squeeze(1)
            
            focal_weights = self.alpha * (1 - prob_true)**self.gamma
            focal_loss = focal_weights * ce_loss
            
            # Combine CE loss and Focal loss
            mixed_loss = ce_loss + self.focal_weight*focal_loss
            
            return mixed_loss.mean()

class DSKDLoss(nn.Module):
    def __init__(self, name, use_this, alpha=0.2, beta=0.02, mu=0.005, num_classes=1000):
        super(DSKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.num_classes = num_classes
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.label_smooth_loss = LabelSmoothingLoss(num_classes=num_classes, alpha=alpha, gamma=2)

    def forward_simple(self, logit_s, gt_label, epsilon=0.1):
        m = len(gt_label)
        label = gt_label.view(len(gt_label), -1)
        value = torch.ones_like(label)
        labels_one_hot = F.one_hot(gt_label, num_classes=self.num_classes).float()
        
        N, c = logit_s.shape
        
        # final logit
        s_i = F.softmax(logit_s, dim=1)
        s_t = torch.gather(s_i, 1, label)

        # soft target label
        p_t = s_t ** 2
        p_t = p_t + value - p_t.mean(0, keepdim=True)
        p_t[value == 0] = 0
        p_t = p_t.detach()

        s_i = self.log_softmax(logit_s)
        s_t = torch.gather(s_i, 1, label)
        loss_t = - (p_t * s_t).sum(dim=1).mean()

        inverse_mask = (1 - labels_one_hot)
        inverse_label = inverse_mask.view(len(inverse_mask), -1).long()
        s_i_inverse = torch.gather(s_i, 1, inverse_label)
        p_inverse = s_i_inverse ** 2
        p_inverse = p_inverse - p_inverse.mean(0, keepdim=True)
        p_inverse = p_inverse.detach()
        log_s_i_inverse = torch.gather(s_i, 1, inverse_label)
        loss_ut = - (p_inverse * log_s_i_inverse).sum(dim=1).mean()

        loss = self.alpha * loss_t + self.beta * loss_ut 

        # 添加 NaN 检查
        if torch.isnan(loss).any():
            print("NaN detected in loss calculation")
            print(f"logit_s: {logit_s}")
            print(f"s_t: {s_t}")
            print(f"p_t: {p_t}")
            print(f"loss_t: {loss_t}")
            print(f"loss_ut: {loss_ut}")

        return loss

    def forward(self, logit_s, gt_label,weight=None):
        if isinstance(logit_s, list):
            if weight==None:
                weight = [1] * len(logit_s)
            else:
                if len(logit_s) != len(weight):
                    if len(logit_s) != weight.shape[1]:
                        raise ValueError("Number of logits and weights must match")
                    else:
                        weight = weight.mean(dim=0)
                        # 将 tensor 转换为列表
                        weight = weight.tolist()
            total_loss = 0.0
            for i in range(len(logit_s)):
                total_loss += weight[i] * self.forward_simple(logit_s[i], gt_label)
            return total_loss / len(logit_s)
        else:
            return self.forward_simple(logit_s, gt_label)

class AJS_loss(torch.nn.Module):
    def __init__(self, num_classes, weight_target, weights):
        super(AJS_loss, self).__init__()
        self.num_classes = num_classes
        self.weight_target = weight_target
        self.weights = [weight_target] + [float(w) for w in weights]
        scaled = True
        if scaled:
            self.scale = -1.0 /  ((1.0 - self.weights[0]) * np.log((1.0 - self.weights[0])))
        else:
            self.scale = 1.0

    def forward(self, pred, labels, weight=None):
        preds = list()
        if isinstance(pred, list):
            for p in pred:
                preds.append(torch.sigmoid(p))
        else:
            preds.append(torch.sigmoid(pred))
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes)
        distribs = [labels_onehot] + preds
        assert len(self.weights) == len(distribs)
        mean_distrib = sum([w * d for w, d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()
        jsw = sum([w * custom_kl_div(mean_distrib_log, d) for w, d in zip(self.weights, distribs)])
        return self.scale * jsw

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, alpha, gamma, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param num_classes: number of classes
        :param args: additional arguments containing alpha and gamma for focal loss
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.alpha = alpha
        self.gamma = gamma

    def label_smoothing(self, inputs):
        """
        Apply label smoothing.
        :param inputs: one-hot encoded labels
        :return: smoothed labels
        """
        K = inputs.size(-1)  # Number of classes
        return ((1 - self.smoothing) * inputs) + (self.smoothing / K)

    def compute_loss(self, x, target):
        """
        Compute the Label Smoothing and Focal Loss.
        :param x: model outputs (logits)
        :param target: target labels
        :return: combined loss
        """
        # Compute log softmax of model outputs
        logprobs = F.log_softmax(x, dim=-1)
        # Original target labels in one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        # Apply label smoothing
        smoothed_target = self.label_smoothing(target_one_hot)
        
        # Compute the negative log likelihood loss
        nll_loss = -torch.sum(smoothed_target * logprobs, dim=-1)
        
        # Return the combined loss
        return nll_loss.mean()

    def forward(self, x, target):
        """
        Forward pass for label smoothing loss.
        :param x: model outputs (logits) or list of logits
        :param target: target labels
        :return: combined loss
        """
        if isinstance(x, list):
            total_loss = 0.0
            for logits in x:
                total_loss += self.compute_loss(logits, target)
            return total_loss / len(x)
        else:
            return self.compute_loss(x, target)
class CombinedLoss(nn.Module):
    def __init__(self, loss_functions, weights):
        super(CombinedLoss, self).__init__()
        self.loss_functions = loss_functions
        self.weights = weights

    def forward(self, *inputs):
        total_loss = 0.0
        for loss_fn, weight in zip(self.loss_functions, self.weights):
            total_loss += weight * loss_fn(*inputs)
        return total_loss
# Define build_loss function
def build_loss(loss_name, args):
    # 参数验证
    valid_loss_names = {
        "ce", "ajs", "uskd", "mixed", "mse", "uskd_mixed", "ajs_mixed", "ajs_uskd", "ajs_uskd_mixed"
    }
    if loss_name not in valid_loss_names:
        raise ValueError(f"Invalid loss name: {loss_name}. Valid options are: {', '.join(valid_loss_names)}")

    def create_combined_loss(loss_functions, weights):
        if len(weights) != len(loss_functions):
            raise ValueError(f"The number of weights must match the number of loss functions ({len(loss_functions)})")
        return CombinedLoss(loss_functions=loss_functions, weights=weights)

    def create_initial_weights(weight_target, n_ensemble):
        return [(1 - weight_target) / n_ensemble for _ in range(n_ensemble)]

    if loss_name == "ce":
        loss = ce_loss
    elif loss_name == "ajs":
        weight_target = args.loss_ajs_weight_target
        n_ensemble = args.num_experts
        initial_weights = create_initial_weights(weight_target, n_ensemble)
        loss = AJS_loss(num_classes=args.num_class, weight_target=weight_target, weights=initial_weights)
    elif loss_name == "uskd":
        loss = DSKDLoss(name="uskd_loss", use_this=True, num_classes=args.num_class)
    elif loss_name == "mixed":
        loss = MixedLoss(alpha=args.alpha, gamma=args.gamma, focal_weight=args.focal_weight)
    elif loss_name == "mse":
        loss = nn.MSELoss()
    elif loss_name == "uskd_mixed":
        loss_functions = [
            DSKDLoss(name="uskd_loss", use_this=True, num_classes=args.num_class),
            MixedLoss(alpha=args.alpha, gamma=args.gamma, focal_weight=args.focal_weight)
        ]
        loss = create_combined_loss(loss_functions, args.weights)
    elif loss_name == "ajs_mixed":
        weight_target = args.loss_ajs_weight_target
        n_ensemble = args.num_experts
        initial_weights = create_initial_weights(weight_target, n_ensemble)
        loss_functions = [
            AJS_loss(num_classes=args.num_class, weight_target=weight_target, weights=initial_weights),
            MixedLoss(alpha=args.alpha, gamma=args.gamma, focal_weight=args.focal_weight)
        ]
        loss = create_combined_loss(loss_functions, args.weights)
    elif loss_name == "ajs_uskd":
        weight_target = args.loss_ajs_weight_target
        n_ensemble = args.num_experts
        initial_weights = create_initial_weights(weight_target, n_ensemble)
        loss_functions = [
            AJS_loss(num_classes=args.num_class, weight_target=weight_target, weights=initial_weights),
            DSKDLoss(name="uskd_loss", use_this=True, num_classes=args.num_class)
        ]
        loss = create_combined_loss(loss_functions, args.weights)
    elif loss_name == "ajs_uskd_mixed":
        weight_target = args.loss_ajs_weight_target
        n_ensemble = args.num_experts
        initial_weights = create_initial_weights(weight_target, n_ensemble)
        loss_functions = [
            AJS_loss(num_classes=args.num_class, weight_target=weight_target, weights=initial_weights),
            DSKDLoss(name="uskd_loss", use_this=True, num_classes=args.num_class),
            MixedLoss(alpha=args.alpha, gamma=args.gamma, focal_weight=args.focal_weight)
        ]
        loss = create_combined_loss(loss_functions, args.weights)

    return loss
def Diversity_loss(logits, targets):
    """
    logits: Tensor of shape [batch_size, num_experts, num_classes]
    targets: Tensor of shape [batch_size, num_classes]
    """
    batch_size, num_experts, num_classes = logits.shape
    
    # Initialize the loss
    loss = 0.0
    
    # Softmax over the classes dimension
    s_i = F.softmax(logits, dim=2)
    
    # Loop over each expert
    for i in range(num_experts):
        logit_s = logits[:, i, :]  # Shape: [batch_size, num_classes]
        
        # Gather the logits corresponding to the target classes
        if targets.dim() == 1:
            # 如果维度为1，使用 unsqueeze 在第1维增加一个维度
            target_indices = targets.unsqueeze(1)
        else:
            # 如果维度不是1，使用本身即可
            target_indices = targets
        s_t = torch.gather(s_i[:, i, :], 1, target_indices)  # Shape: [batch_size, 1]
        
        # Calculate the square of the gathered logits
        p_t = s_t ** 2  # Shape: [batch_size, 1]
        
        # Sum the squares of other experts' logits
        s_other_2 = s_i[:, torch.arange(num_experts) != i, :]** 2
        other_experts_sum = torch.sum(s_i[:, torch.arange(num_experts) != i, :]** 2, dim=1)   # Shape: [batch_size, num_classes]
        
        # Mean of the summed logits' squares across the batch
        other_experts_mean = other_experts_sum.mean(dim=0, keepdim=True)  # Shape: [1, num_classes]
        other_experts_mean_expanded = other_experts_mean.expand(batch_size, -1)

        # Generate the soft target labels
        soft_target = 1 - other_experts_sum - other_experts_mean_expanded
        
        # Apply the mask where target is 0
        soft_target[targets == 0] = 0
        soft_target = soft_target.detach()  # Detach to prevent gradient flow
        
        # Compute the log softmax for the current expert's logits
        s_i_log_softmax = F.log_softmax(logit_s, dim=1)
        
        # Gather the log softmax values corresponding to the target classes
        s_t_log_softmax = torch.gather(s_i_log_softmax, 1, target_indices)  # Shape: [batch_size, 1]
        
        # Compute the KL divergence loss
        loss_i = - (soft_target * s_t_log_softmax).sum(dim=1).mean()  # Shape: scalar
        
        # Accumulate the loss
        loss += loss_i
    
    # Return the average loss across all experts
    return loss / num_experts

