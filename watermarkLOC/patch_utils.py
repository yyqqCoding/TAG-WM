"""
Patch Processing Utilities for SEAL-LOC

包含补丁级别的处理函数，包括：
- L2距离计算
- 图像变换
- 补丁映射和处理工具
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List


def transform_img(image: Image.Image, target_size: int = 512) -> torch.Tensor:
    """
    图像预处理变换，将PIL图像转换为模型输入格式
    
    从baseline中缺失的函数，用于将图像转换为张量格式
    
    Args:
        image: PIL图像
        target_size: 目标尺寸 (默认512)
        
    Returns:
        预处理后的图像张量 (C, H, W)
    """
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
    ])
    
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image")
    
    # 确保是RGB格式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    tensor = transform(image)
    return tensor


def calculate_patch_l2(noise1: torch.Tensor, noise2: torch.Tensor, k: int) -> float:
    """
    计算两个噪声张量在patch级别的最小L2距离
    
    这是baseline中缺失的关键函数，用于SEAL方案的水印检测
    
    Args:
        noise1: 第一个噪声张量 (1, 4, 64, 64)
        noise2: 第二个噪声张量 (1, 4, 64, 64) 
        k: patch数量 (必须是完全平方数)
        
    Returns:
        最小L2距离值
    """
    # 确保k是完全平方数
    patch_per_side = int(np.sqrt(k))
    assert patch_per_side ** 2 == k, f"k={k} must be a perfect square"
    
    # 获取噪声张量的形状
    if noise1.dim() == 4:
        _, c, h, w = noise1.shape
        noise1 = noise1.squeeze(0)  # 移除batch维度
    else:
        c, h, w = noise1.shape
    
    if noise2.dim() == 4:
        noise2 = noise2.squeeze(0)
    
    # 计算每个patch的大小
    patch_h = h // patch_per_side
    patch_w = w // patch_per_side
    
    min_l2_distances = []
    
    # 遍历每个patch
    for i in range(patch_per_side):
        for j in range(patch_per_side):
            # 计算patch的边界
            y_start = i * patch_h
            y_end = (i + 1) * patch_h
            x_start = j * patch_w
            x_end = (j + 1) * patch_w
            
            # 提取对应的patch
            patch1 = noise1[:, y_start:y_end, x_start:x_end]
            patch2 = noise2[:, y_start:y_end, x_start:x_end]
            
            # 计算L2距离
            l2_distance = torch.norm(patch1 - patch2, p=2).item()
            min_l2_distances.append(l2_distance)
    
    # 返回最小L2距离
    return min(min_l2_distances)


def calculate_patch_l2_all(noise1: torch.Tensor, noise2: torch.Tensor, k: int) -> List[float]:
    """
    计算所有patch的L2距离
    
    Args:
        noise1: 第一个噪声张量
        noise2: 第二个噪声张量
        k: patch数量
        
    Returns:
        所有patch的L2距离列表
    """
    patch_per_side = int(np.sqrt(k))
    assert patch_per_side ** 2 == k, f"k={k} must be a perfect square"
    
    if noise1.dim() == 4:
        noise1 = noise1.squeeze(0)
    if noise2.dim() == 4:
        noise2 = noise2.squeeze(0)
        
    c, h, w = noise1.shape
    patch_h = h // patch_per_side
    patch_w = w // patch_per_side
    
    l2_distances = []
    
    for i in range(patch_per_side):
        for j in range(patch_per_side):
            y_start = i * patch_h
            y_end = (i + 1) * patch_h
            x_start = j * patch_w
            x_end = (j + 1) * patch_w
            
            patch1 = noise1[:, y_start:y_end, x_start:x_end]
            patch2 = noise2[:, y_start:y_end, x_start:x_end]
            
            l2_distance = torch.norm(patch1 - patch2, p=2).item()
            l2_distances.append(l2_distance)
    
    return l2_distances


def map_latent_to_image_coords(patch_idx: int, patch_grid_size: int, 
                              latent_size: Tuple[int, int], image_size: int = 512) -> Tuple[int, int, int, int]:
    """
    将潜空间patch索引映射到图像空间坐标
    
    Args:
        patch_idx: patch索引 (0到patch_grid_size²-1)
        patch_grid_size: 网格大小 (如8表示8x8网格)
        latent_size: 潜空间尺寸 (H, W)
        image_size: 图像尺寸 (默认512)
        
    Returns:
        图像空间的坐标 (x_start, y_start, x_end, y_end)
    """
    # 计算patch在网格中的位置
    i = patch_idx // patch_grid_size  # 行
    j = patch_idx % patch_grid_size   # 列
    
    # 计算图像空间中patch的大小
    patch_size_image = image_size // patch_grid_size
    
    # 计算坐标
    y_start = i * patch_size_image
    y_end = (i + 1) * patch_size_image
    x_start = j * patch_size_image
    x_end = (j + 1) * patch_size_image
    
    return x_start, y_start, x_end, y_end


def extract_patch_from_image(image: Image.Image, patch_idx: int, patch_grid_size: int) -> Image.Image:
    """
    从图像中提取指定的patch
    
    Args:
        image: 原始图像 (PIL Image)
        patch_idx: patch索引
        patch_grid_size: 网格大小
        
    Returns:
        提取的patch图像
    """
    x_start, y_start, x_end, y_end = map_latent_to_image_coords(
        patch_idx, patch_grid_size, (64, 64), image.size[0]
    )
    
    patch_image = image.crop((x_start, y_start, x_end, y_end))
    return patch_image


def visualize_patch_grid(image: Image.Image, patch_grid_size: int = 8, 
                        line_color: str = 'red', line_width: int = 2) -> Image.Image:
    """
    在图像上可视化patch网格
    
    Args:
        image: 原始图像
        patch_grid_size: 网格大小
        line_color: 网格线颜色
        line_width: 网格线宽度
        
    Returns:
        带有网格线的图像
    """
    from PIL import ImageDraw
    
    # 创建图像副本
    image_with_grid = image.copy()
    draw = ImageDraw.Draw(image_with_grid)
    
    width, height = image.size
    patch_w = width // patch_grid_size
    patch_h = height // patch_grid_size
    
    # 绘制垂直线
    for i in range(1, patch_grid_size):
        x = i * patch_w
        draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)
    
    # 绘制水平线
    for i in range(1, patch_grid_size):
        y = i * patch_h
        draw.line([(0, y), (width, y)], fill=line_color, width=line_width)
    
    return image_with_grid


def compute_patch_statistics(tensor: torch.Tensor, k: int) -> dict:
    """
    计算patch级别的统计信息
    
    Args:
        tensor: 输入张量 (C, H, W)
        k: patch数量
        
    Returns:
        包含统计信息的字典
    """
    patch_per_side = int(np.sqrt(k))
    assert patch_per_side ** 2 == k
    
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    c, h, w = tensor.shape
    patch_h = h // patch_per_side
    patch_w = w // patch_per_side
    
    stats = {
        'patch_means': [],
        'patch_stds': [],
        'patch_mins': [],
        'patch_maxs': []
    }
    
    for i in range(patch_per_side):
        for j in range(patch_per_side):
            y_start = i * patch_h
            y_end = (i + 1) * patch_h
            x_start = j * patch_w
            x_end = (j + 1) * patch_w
            
            patch = tensor[:, y_start:y_end, x_start:x_end]
            
            stats['patch_means'].append(torch.mean(patch).item())
            stats['patch_stds'].append(torch.std(patch).item())
            stats['patch_mins'].append(torch.min(patch).item())
            stats['patch_maxs'].append(torch.max(patch).item())
    
    return stats


if __name__ == "__main__":
    # 测试patch工具函数
    print("Testing patch utilities...")
    
    # 创建测试数据
    test_noise1 = torch.randn(1, 4, 64, 64)
    test_noise2 = torch.randn(1, 4, 64, 64)
    
    # 测试L2距离计算
    min_l2 = calculate_patch_l2(test_noise1, test_noise2, k=64)
    all_l2 = calculate_patch_l2_all(test_noise1, test_noise2, k=64)
    
    print(f"Minimum L2 distance: {min_l2:.4f}")
    print(f"Number of patch L2 distances: {len(all_l2)}")
    print(f"Average L2 distance: {np.mean(all_l2):.4f}")
    
    # 测试坐标映射
    x_start, y_start, x_end, y_end = map_latent_to_image_coords(0, 8, (64, 64))
    print(f"Patch 0 coordinates: ({x_start}, {y_start}, {x_end}, {y_end})")
    
    # 测试统计信息
    stats = compute_patch_statistics(test_noise1, k=64)
    print(f"Patch statistics computed for {len(stats['patch_means'])} patches")
    
    print("Patch utilities test completed!") 