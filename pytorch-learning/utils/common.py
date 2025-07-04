"""
PyTorch 学习通用工具函数
"""
import torch
import os
import random
import numpy as np


def get_device():
    """
    获取可用的设备 (CUDA/MPS/CPU)
    
    Returns:
        torch.device: 可用的设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 使用 CUDA: {torch.cuda.get_device_name()}")
        print(f"📊 CUDA 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 使用 Apple MPS")
    else:
        device = torch.device("cpu")
        print("💻 使用 CPU")
    
    return device


def set_seed(seed=42):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 设置随机种子: {seed}")


def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model (torch.nn.Module): PyTorch 模型
        
    Returns:
        int: 可训练参数总数
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    return trainable_params


def create_dirs(*dirs):
    """
    创建必要的目录
    
    Args:
        *dirs: 需要创建的目录路径
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 目录已创建: {dir_path}")


def print_tensor_info(tensor, name="Tensor"):
    """
    打印张量信息
    
    Args:
        tensor (torch.Tensor): 要分析的张量
        name (str): 张量名称
    """
    print(f"📊 {name} 信息:")
    print(f"  形状: {tensor.shape}")
    print(f"  数据类型: {tensor.dtype}")
    print(f"  设备: {tensor.device}")
    print(f"  内存占用: {tensor.element_size() * tensor.nelement() / 1e6:.2f} MB")
    if tensor.numel() <= 20:  # 只有小张量才打印值
        print(f"  值: {tensor}")
    else:
        print(f"  值范围: [{tensor.min():.4f}, {tensor.max():.4f}]")


def format_time(seconds):
    """
    格式化时间显示
    
    Args:
        seconds (float): 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"