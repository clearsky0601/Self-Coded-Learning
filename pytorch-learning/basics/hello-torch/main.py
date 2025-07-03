#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hello PyTorch 练习
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.common import get_device

def main():
    """主函数"""
    print("🧠 Hello PyTorch 练习开始!")
    
    # 检查设备
    device = get_device()
    
    # 创建张量
    x = torch.randn(3, 4)
    y = torch.randn(4, 2)
    
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    
    # 矩阵乘法
    z = torch.matmul(x, y)
    print(f"z = x @ y, shape: {z.shape}")
    
    # 移动到GPU（如果可用）
    if device.type == 'cuda':
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        z_gpu = torch.matmul(x_gpu, y_gpu)
        print(f"GPU计算完成，结果形状: {z_gpu.shape}")
        print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    print("✅ 练习完成!")

if __name__ == "__main__":
    main()