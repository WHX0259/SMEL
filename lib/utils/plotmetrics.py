import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    f1_score,

)
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from lib.utils.metrics import analysis_pred_binary
def plot_all_experts_tsne(train_features, train_labels, val_features, val_labels, num_classes, experts_range, filename='tsne_plot.png'):
    # Generate a colormap for different experts
    num_experts = len(experts_range)
    cmap = plt.get_cmap('tab10')  # Use a colormap that supports many colors
    # colors = [cmap(i) for i in np.linspace(0, 1, num_experts)]  # Generate colors for each expert
    colors = [
          "#5285c6","#3fa0c0","#4c6c43","#d6e0c8","#b55489","#f1a19a"
        ]
    # Combine training and validation features
    all_features = [np.concatenate([train_features[:, expert_idx, :], val_features[:, expert_idx, :]]) for expert_idx in experts_range]
    
    # Initialize TSNE
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, n_iter=1000)
    
    # Transform all features using TSNE
    tsne_results = [tsne.fit_transform(features) for features in all_features]
    
    plt.figure(figsize=(10, 10))
    
    for expert_idx, tsne_result in enumerate(tsne_results):
        train_tsne_results = tsne_result[:len(train_features)]
        val_tsne_results = tsne_result[len(train_features):]
        
        for class_idx in range(num_classes):
            train_indices = train_labels == class_idx
            val_indices = val_labels == class_idx
            
            plt.scatter(train_tsne_results[train_indices, 0], train_tsne_results[train_indices, 1],
                        label=f'Train Class {class_idx} (Expert {expert_idx})', alpha=0.5, color=colors[expert_idx*num_classes+class_idx])
            
            plt.scatter(val_tsne_results[val_indices, 0], val_tsne_results[val_indices, 1],
                        label=f'Val Class {class_idx} (Expert {expert_idx})', alpha=0.8, edgecolor='k', color=colors[expert_idx*num_classes+class_idx])
    
    plt.legend()
    plt.title("t-SNE of Features from All Experts")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_tsne(train_features, train_labels, val_features, val_labels, num_classes, filename='tsne_plot.png'):
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, n_iter=1000)
    all_features = np.concatenate([train_features, val_features])
    
    # Check if all_features is valid
    if all_features.shape[1] == 0:
        raise ValueError("The input features for TSNE are empty.")
    
    tsne_results = tsne.fit_transform(all_features)
    
    train_tsne_results = tsne_results[:len(train_features)]
    val_tsne_results = tsne_results[len(train_features):]
    
    # Define the colors: pink and blue
    colors = ['#FF69B4', '#1E90FF']  # Pink and blue

    plt.figure(figsize=(10, 10))
    for class_idx in range(num_classes):
        train_indices = train_labels == class_idx
        val_indices = val_labels == class_idx
        
        plt.scatter(train_tsne_results[train_indices, 0], train_tsne_results[train_indices, 1], 
                    label=f'Train Class {class_idx}', alpha=0.5, color=colors[class_idx % len(colors)])
        
        plt.scatter(val_tsne_results[val_indices, 0], val_tsne_results[val_indices, 1], 
                    label=f'Val Class {class_idx}', alpha=0.8, edgecolor='k', color=colors[class_idx % len(colors)])
    
    plt.legend()
    plt.title("t-SNE of Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_losses(train_losses, val_losses1, val_losses2, output_path):
    """
    绘制训练集和两个测试集（验证集和EMA验证集）的损失曲线，并保存为图片。

    Args:
    - train_losses (list): 训练集损失列表
    - val_losses1 (list): 验证集1损失列表
    - val_losses2 (list): EMA验证集损失列表
    - output_path (str): 图片保存路径
    """

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_path, exist_ok=True)

    # 绘制训练集损失图
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'train_loss_curve.png'))
    plt.show()

    # 绘制验证集1损失图
    plt.figure()
    plt.plot(range(1, len(val_losses1) + 1), val_losses1, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'val_loss_curve1.png'))
    plt.show()

    # 绘制EMA验证集损失图
    plt.figure()
    plt.plot(range(1, len(val_losses2) + 1), val_losses2, label='EMA Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('EMA Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'val_loss_curve2.png'))
    plt.show()
def plot_multiclass_roc_curve(targets, all_outputs, method, fold, output):
    plt.figure(figsize=(12, 8))
    n_experts = all_outputs.shape[1]
    n_classes = all_outputs.shape[2]
    
    for expert_idx in range(n_experts):
        for class_idx in range(n_classes):
            fpr, tpr, _ = roc_curve(targets, all_outputs[:, expert_idx, class_idx], pos_label=class_idx)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f}) for Expert {expert_idx + 1}, Class {class_idx}')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {method} - Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output, f'ROC_{method}_fold_{fold}.png'))
    plt.show()

def plot_multiclass_pr_curve(targets, all_outputs, method, fold, output):
    plt.figure(figsize=(12, 8))
    n_experts = all_outputs.shape[1]
    n_classes = all_outputs.shape[2]
    
    for expert_idx in range(n_experts):
        for class_idx in range(n_classes):
            precision, recall, _ = precision_recall_curve(targets, all_outputs[:, expert_idx, class_idx], pos_label=class_idx)
            plt.plot(recall, precision, label=f'PR Curve for Expert {expert_idx + 1}, Class {class_idx}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {method} - Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output, f'PR_{method}_fold_{fold}.png'))
    plt.show()
def plot_dca_curves(targets, dl_outputs, filename):
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefit = np.zeros(len(thresholds))
    
    for j, threshold in enumerate(thresholds):
        y_pred = (dl_outputs[:, 1] >= threshold).astype(int)  # 假设二分类模型
        tn, fp, fn, tp = confusion_matrix(targets, y_pred).ravel()
        
        n = len(targets)
        net_benefit[j] = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, net_benefit, label='Net Benefit')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_roc_curve(y_true, y_scores, num_classes, filename):
    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()

def plot_pr_curve(y_true, y_scores, num_classes, filename):
    plt.figure()
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        average_precision = average_precision_score(y_true[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {average_precision:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(filename)
    plt.show()
