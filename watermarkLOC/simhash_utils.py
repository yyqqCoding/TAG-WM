"""
SimHash Utilities for SEAL-LOC

从baseline/caption_pairs.py适配的SimHash函数，
专门优化用于SEAL-LOC的逐patch语义哈希计算。
"""

import torch
import numpy as np
import zlib
import random
from typing import List


def simhash_single_patch(embedding: torch.Tensor, num_bits: int = 7, seed: int = 42) -> int:
    """
    为单个patch计算SimHash值
    
    这是从baseline的compute_simhash函数适配而来，
    但专门为单个语义向量设计，去除了多patch循环。
    
    Args:
        embedding: 单个patch的语义向量 (768维)
        num_bits: SimHash比特数 (默认7)
        seed: 随机种子，确保可重现性
        
    Returns:
        整数哈希值
    """
    random.seed(seed)
    
    # 确保输入是CPU上的numpy数组
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu()
    
    bits = [0] * num_bits
    
    # 生成num_bits个随机向量，与embedding做点积
    for bit_index in range(num_bits):
        random_vector = torch.randn_like(embedding)
        dot_product = torch.dot(random_vector, embedding)
        bits[bit_index] = 1 if dot_product > 0 else 0
    
    # 使用CRC32生成最终哈希值
    hash_value = zlib.crc32(bytes(bits)) & 0xFFFFFFFF
    
    return hash_value


def simhash_batch_patches(embeddings: List[torch.Tensor], num_bits: int = 7, seed: int = 42) -> List[int]:
    """
    批量计算多个patch的SimHash值
    
    Args:
        embeddings: 多个patch的语义向量列表
        num_bits: SimHash比特数
        seed: 随机种子
        
    Returns:
        哈希值列表
    """
    hash_values = []
    
    for i, embedding in enumerate(embeddings):
        # 为每个patch使用不同的种子，确保独立性
        patch_seed = seed + i
        hash_value = simhash_single_patch(embedding, num_bits, patch_seed)
        hash_values.append(hash_value)
    
    return hash_values


def compute_simhash_compatibility(embedding, num_patches, num_bits, seed):
    """
    兼容baseline的compute_simhash函数接口
    
    保持与原始函数相同的签名，但内部实现优化
    """
    if num_patches == 1:
        # 单patch情况，直接调用优化版本
        return [simhash_single_patch(embedding, num_bits, seed)]
    else:
        # 多patch情况，使用原始逻辑
        return _original_compute_simhash(embedding, num_patches, num_bits, seed)


def _original_compute_simhash(embedding, num_patches, num_bits, seed):
    """
    原始的compute_simhash实现，直接从baseline复制
    """
    random.seed(seed)
    hash_keys = []
    
    for patch_index in range(num_patches):
        bits = [0] * num_bits
        for bit_index in range(num_bits):
            random_vector = torch.randn_like(embedding)
            bits[bit_index] = 1 if torch.dot(random_vector, embedding) > 0 else 0
            bits[bit_index] = (bits[bit_index] + bit_index + patch_index) % 256
        hash_keys.append(zlib.crc32(bytes(bits)) & 0xFFFFFFFF)
    
    return hash_keys


def verify_simhash_consistency(embedding: torch.Tensor, num_bits: int = 7, seed: int = 42, iterations: int = 5) -> bool:
    """
    验证SimHash的一致性
    
    多次计算同一个embedding的SimHash，确保结果一致
    
    Args:
        embedding: 语义向量
        num_bits: SimHash比特数
        seed: 随机种子
        iterations: 测试迭代次数
        
    Returns:
        是否所有迭代结果都一致
    """
    first_hash = simhash_single_patch(embedding, num_bits, seed)
    
    for _ in range(iterations - 1):
        current_hash = simhash_single_patch(embedding, num_bits, seed)
        if current_hash != first_hash:
            return False
    
    return True


if __name__ == "__main__":
    # 测试SimHash函数
    print("Testing SimHash utilities...")
    
    # 创建测试语义向量
    test_embedding = torch.randn(768)
    
    # 测试单patch SimHash
    hash1 = simhash_single_patch(test_embedding, num_bits=7, seed=42)
    hash2 = simhash_single_patch(test_embedding, num_bits=7, seed=42)
    
    print(f"Hash consistency test: {hash1 == hash2}")
    print(f"Generated hash: {hash1}")
    
    # 测试一致性
    is_consistent = verify_simhash_consistency(test_embedding)
    print(f"SimHash consistency over multiple iterations: {is_consistent}")
    
    # 测试批量处理
    test_embeddings = [torch.randn(768) for _ in range(5)]
    batch_hashes = simhash_batch_patches(test_embeddings)
    print(f"Batch hash generation: {len(batch_hashes)} hashes generated")
    
    print("SimHash utilities test completed!") 