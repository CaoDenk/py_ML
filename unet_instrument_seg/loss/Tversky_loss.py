import torch
import torch.nn as nn


import torch

def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-6):
    # 计算交集和差集
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1 - y_true) * y_pred)
    fn = torch.sum(y_true * (1 - y_pred))
    
    # 计算 Tversky 指数
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    
    # 计算 Tversky loss
    loss = 1 - tversky
    
    return loss