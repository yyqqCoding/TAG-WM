"""
SEAL-LOC: Semantic-Aware Localization Watermark Embedder

基于语义的定位水印嵌入器，集成SEAL的语义感知能力与TAG-WM的双水印架构。
核心特性：
- 继承TAG-WM的双水印架构和DMJS算法
- 实现逐补丁语义特征提取
- 动态生成与内容绑定的语义定位水印
- 保持与原有系统的完全兼容性
"""

import torch
import torch.nn as nn
import numpy as np
import os
import math
import hashlib
from typing import Tuple, List, Optional
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from model_config import get_model_path, check_model_exists

# 导入TAG-WM的核心组件
import sys
sys.path.append('../applied_to_sd2')

# 临时禁用DVRD导入以避免依赖问题
import os
os.environ['DISABLE_DVRD'] = '1'
from watermark_embedder import WatermarkEmbedder

# 导入本地的图像描述生成函数
from caption_utils import generate_caption


class SEALLOCEmbedder(WatermarkEmbedder):
    """
    语义区域感知定位水印嵌入器 (SEAL-LOC)
    
    核心创新：
    - 将定位水印从固定模板(TLT)升级为动态语义绑定模板(W_loc^S)
    - 每个patch的水印模式与其语义内容密码学级别绑定
    - 保持TAG-WM的版权水印(W_cop)和DMJS采样机制不变
    """
    
    def __init__(self, 
                 # 继承TAG-WM的所有参数
                 wm_len=256,
                 center_interval_ratio=0.5,
                 shuffle_random_seed=133563,
                 encrypt_random_seed=133563,
                 tlt_intervals_num=3,  # 使用三区间策略
                 fpr=1e-6,
                 user_number=1000000,
                 optimize_tamper_loc_method=None,
                 DVRD_checkpoint_path=None,
                 DVRD_train_size=512,
                 device='cuda',
                 # SEAL-LOC特有参数
                 patch_grid_size=8,  # 8x8网格，共64个patch
                 semantic_vector_dim=768,  # SentenceTransformer输出维度
                 simhash_bits=7,  # SimHash比特数
                 semantic_maps_dir='watermarkLOC/semantic_maps/',  # 语义地图存储路径
                 vlm_model_name='blip2-flan-t5-xl',  # VLM模型
                 sentence_model_name='kasraarabi/finetuned-caption-embedding',  # 语义编码模型
                 ):
        
        # 初始化父类
        super(SEALLOCEmbedder, self).__init__(
            wm_len=wm_len,
            center_interval_ratio=center_interval_ratio,
            shuffle_random_seed=shuffle_random_seed,
            encrypt_random_seed=encrypt_random_seed,
            tlt_intervals_num=tlt_intervals_num,
            fpr=fpr,
            user_number=user_number,
            optimize_tamper_loc_method=optimize_tamper_loc_method,
            DVRD_checkpoint_path=DVRD_checkpoint_path,
            DVRD_train_size=DVRD_train_size,
            device=device
        )
        
        # SEAL-LOC特有配置
        self.patch_grid_size = patch_grid_size
        self.num_patches = patch_grid_size * patch_grid_size  # 64个patch
        self.semantic_vector_dim = semantic_vector_dim
        self.simhash_bits = simhash_bits
        self.semantic_maps_dir = semantic_maps_dir
        
        # 确保存储目录存在
        os.makedirs(semantic_maps_dir, exist_ok=True)
        
        # 初始化VLM和语义编码模型
        model_path = get_model_path(vlm_model_name)
        print(f"  - Loading VLM model: {vlm_model_name} from {model_path}")
        
        # 检查模型路径是否存在
        if not check_model_exists(model_path):
            print(f"  ⚠️ Model path does not exist: {model_path}")
            print("  请确保已将模型下载到指定路径")
        
        # 尝试解决BLIP-2 tokenizer问题的多种方法
        try:
            # 方法1: 从本地路径加载
            self.vlm_processor = Blip2Processor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            self.vlm_model = Blip2ForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True
            ).to(device)
            print("  ✅ VLM loaded successfully from local path")
        except Exception as e1:
            print(f"  ⚠️ Local loading failed: {e1}")
            try:
                # 方法2: 使用较小的BLIP-2模型
                fallback_model_path = get_model_path("blip2-opt-2.7b")
                print(f"  🔄 Trying fallback model: {fallback_model_path}")
                self.vlm_processor = Blip2Processor.from_pretrained(
                    fallback_model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
                self.vlm_model = Blip2ForConditionalGeneration.from_pretrained(
                    fallback_model_path, 
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    local_files_only=True
                ).to(device)
                print("  ✅ VLM loaded successfully (fallback model)")
            except Exception as e2:
                print(f"  ❌ All VLM loading methods failed: {e2}")
                print("  请确保BLIP-2模型已正确下载到本地路径")
                raise RuntimeError("无法加载BLIP-2模型，请检查模型文件是否存在")
        
        # 初始化SentenceTransformer（支持本地路径）
        sentence_model_path = get_model_path(sentence_model_name)
        print(f"  - Loading SentenceTransformer: {sentence_model_name} from {sentence_model_path}")
        
        try:
            # 尝试从本地路径加载
            self.sentence_model = SentenceTransformer(sentence_model_path).to(device)
            print("  ✅ SentenceTransformer loaded successfully from local path")
        except Exception as e:
            print(f"  ⚠️ Local SentenceTransformer loading failed: {e}")
            try:
                # 回退到在线加载
                self.sentence_model = SentenceTransformer(sentence_model_name).to(device)
                print("  ✅ SentenceTransformer loaded successfully from online")
            except Exception as e2:
                print(f"  ❌ All SentenceTransformer loading methods failed: {e2}")
                raise RuntimeError("无法加载SentenceTransformer模型")
        
        print(f"SEAL-LOC Embedder initialized:")
        print(f"  - Patch grid: {patch_grid_size}x{patch_grid_size} = {self.num_patches} patches")
        print(f"  - Semantic vector dim: {semantic_vector_dim}")
        print(f"  - SimHash bits: {simhash_bits}")
        print(f"  - Semantic maps storage: {semantic_maps_dir}")
    
    def generate_proxy_image(self, prompt: str, pipe=None, num_inference_steps=20) -> Image.Image:
        """
        阶段一：代理生成与初始表示
        生成无水印的代理图像，用于后续语义提取
        """
        if pipe is not None:
            with torch.no_grad():
                proxy_image = pipe(prompt).images[0]
        else:
            # 使用内置的VLM生成简单的代理图像
            from diffusers import StableDiffusionPipeline
            temp_pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16
            ).to(self.device)
            with torch.no_grad():
                proxy_image = temp_pipe(prompt, num_inference_steps=num_inference_steps).images[0]
            del temp_pipe  # 释放内存
        return proxy_image
    
    def extract_patch_semantics(self, proxy_image: Image.Image, latent_size: Tuple[int, int, int]) -> List[torch.Tensor]:
        """
        阶段二：逐补丁语义特征提取
        
        Args:
            proxy_image: 代理图像 (512x512 RGB)
            latent_size: 潜空间尺寸 (C, H, W) = (4, 64, 64)
            
        Returns:
            List of semantic vectors, one for each patch
        """
        _, latent_h, latent_w = latent_size
        patch_size_latent = latent_h // self.patch_grid_size  # 64 // 8 = 8
        patch_size_image = 512 // self.patch_grid_size  # 512 // 8 = 64
        
        semantic_vectors = []
        
        print(f"Extracting semantics for {self.num_patches} patches...")
        
        for i in range(self.patch_grid_size):
            for j in range(self.patch_grid_size):
                # 计算patch在图像空间的位置
                y_start = i * patch_size_image
                y_end = (i + 1) * patch_size_image
                x_start = j * patch_size_image
                x_end = (j + 1) * patch_size_image
                
                # 裁剪对应的图像区域
                patch_image = proxy_image.crop((x_start, y_start, x_end, y_end))
                
                # 使用VLM生成该patch的描述
                patch_caption = generate_caption(
                    patch_image, 
                    self.vlm_processor, 
                    self.vlm_model, 
                    device=self.device
                )
                
                # 将描述转换为语义向量
                semantic_vector = self.sentence_model.encode(
                    patch_caption, 
                    convert_to_tensor=True
                ).to(self.device)
                
                # 归一化语义向量
                semantic_vector = semantic_vector / torch.norm(semantic_vector)
                
                semantic_vectors.append(semantic_vector)
        
        print(f"Semantic extraction completed. Generated {len(semantic_vectors)} vectors.")
        return semantic_vectors
    
    def generate_dynamic_semantic_watermark(self, semantic_vectors: List[torch.Tensor], latent_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        阶段三：动态语义定位水印生成
        
        为每个patch基于其语义向量生成独立的水印模式，
        然后拼接成完整的W_loc^S
        
        Args:
            semantic_vectors: 64个语义向量的列表
            latent_size: 潜空间尺寸 (4, 64, 64)
            
        Returns:
            W_loc^S: 动态语义定位水印 (torch.Tensor)
        """
        latent_len = np.prod(latent_size)  # 4 * 64 * 64 = 16384
        patch_latent_len = latent_len // self.num_patches  # 16384 // 64 = 256
        
        w_loc_s = torch.zeros(latent_len, device=self.device)
        
        print(f"Generating dynamic semantic watermark for {self.num_patches} patches...")
        
        for patch_idx, semantic_vector in enumerate(semantic_vectors):
            # 使用SimHash生成该patch的哈希值
            hash_value = self._compute_patch_simhash(semantic_vector, patch_idx)
            
            # 基于哈希值生成该patch的水印比特流
            patch_watermark_bits = self._generate_patch_watermark_bits(hash_value, patch_latent_len)
            
            # 将该patch的水印填入对应位置
            start_idx = patch_idx * patch_latent_len
            end_idx = (patch_idx + 1) * patch_latent_len
            w_loc_s[start_idx:end_idx] = torch.from_numpy(patch_watermark_bits).float().to(self.device)
        
        print("Dynamic semantic watermark generation completed.")
        return w_loc_s
    
    def generate_xor_based_semantic_watermark(self, semantic_vectors: List[torch.Tensor], latent_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        XOR基础模式的语义定位水印生成
        
        流程图:
        语义向量 v_i → 语义密钥 K_sem ↘
                                            XOR → 最终定位水印 W_loc_final → DMJS → 水印化噪声 Z_T^w
        基础模式 P_base ↗
        
        步骤：
        1. 生成确定性基础模式 P_base（完美棋盘格）
        2. 生成语义密钥 K_sem（基于patch语义）  
        3. XOR混合生成最终定位水印 W_loc_final
        4. 最终分布校正确保严格0.5分布
        
        Args:
            semantic_vectors: 64个语义向量的列表
            latent_size: 潜空间尺寸 (4, 64, 64)
            
        Returns:
            W_loc_final: XOR混合的最终定位水印 (torch.Tensor)
        """
        print("🔀 生成XOR基础模式的语义定位水印...")
        
        latent_len = np.prod(latent_size)  # 4 * 64 * 64 = 16384
        
        # === 步骤1：生成确定性基础模式 P_base ===
        print("  📐 步骤1：生成基础模式（高频棋盘格）")
        P_base = self._generate_perfect_base_pattern(latent_len)
        
        # === 步骤2：生成内容相关的语义密钥 K_sem ===
        print("  🔑 步骤2：生成语义密钥")
        K_sem = self._generate_semantic_modulation_key(semantic_vectors, latent_len)
        
        # === 步骤3：XOR混合生成最终定位水印 ===
        print("  🔀 步骤3：XOR混合生成最终定位水印")
        W_loc_final = P_base ^ K_sem
        
        # === 步骤4：最终分布校正（关键修复）===
        print("  🔧 步骤4：最终分布校正")
        W_loc_final = self._apply_final_distribution_correction_semantic(W_loc_final, semantic_vectors)
        
        # 验证统计特性
        ones_ratio = (W_loc_final == 1).sum() / len(W_loc_final)
        print(f"✅ XOR语义定位水印生成完成")
        print(f"  📊 基础模式1的比例: {(P_base == 1).sum() / len(P_base):.6f}")
        print(f"  📊 语义密钥1的比例: {(K_sem == 1).sum() / len(K_sem):.6f}")
        print(f"  📊 最终水印1的比例: {ones_ratio:.6f} (目标: 0.500000)")
        print(f"  🎯 完美统计特性 + 语义绑定 = 最优定位水印")
        
        # 转换为torch tensor并reshape
        w_loc_tensor = torch.from_numpy(W_loc_final).float().to(self.device)
        w_loc_tensor = w_loc_tensor.reshape(latent_size)
        
        return w_loc_tensor
    
    def _generate_perfect_base_pattern(self, total_bits: int) -> np.ndarray:
        """
        生成完美的基础模式 P_base
        
        特性：
        - 严格交替的0-1-0-1-0-1...模式
        - 完美的伯努利(0.5)分布：0和1数量严格相等
        - 高频棋盘格结构：对任何平滑/模糊操作最敏感
        - 完全确定性：不依赖任何图像内容
        
        Args:
            total_bits: 总比特数 (16384)
            
        Returns:
            P_base: 基础模式数组 (0,1,0,1,...)
        """
        # 完全复现TAG-WM的奇偶模式
        P_base = np.arange(total_bits) % 2
        
        # 验证完美分布特性
        ones_count = (P_base == 1).sum()
        zeros_count = (P_base == 0).sum()
        ones_ratio = ones_count / total_bits
        
        print(f"    📐 基础模式生成: {total_bits}位")
        print(f"    📊 0的数量: {zeros_count}, 1的数量: {ones_count}")
        print(f"    📊 1的比例: {ones_ratio:.6f} (完美目标: 0.500000)")
        print(f"    🎯 高频棋盘格结构: 对篡改最敏感")
        
        return P_base.astype(np.int32)
    
    def _generate_semantic_modulation_key(self, semantic_vectors: List[torch.Tensor], total_bits: int) -> np.ndarray:
        """
        生成语义调制密钥 K_sem
        
        为每个patch基于其语义向量生成确定性的伪随机密钥，
        然后拼接成完整的K_sem
        **新增分布校正机制，确保严格0.5分布**
        
        Args:
            semantic_vectors: 64个语义向量的列表
            total_bits: 总比特数 (16384)
            
        Returns:
            K_sem: 语义调制密钥 (伪随机，但确定性，严格0.5分布)
        """
        num_patches = len(semantic_vectors)  # 64
        bits_per_patch = total_bits // num_patches  # 256
        
        K_sem = np.zeros(total_bits, dtype=np.int32)
        
        print(f"    🔑 语义密钥生成: {num_patches}个patch, 每个{bits_per_patch}位")
        
        for patch_idx, semantic_vector in enumerate(semantic_vectors):
            # 使用增强的SimHash生成该patch的种子
            patch_seed = self._compute_enhanced_patch_seed(semantic_vector, patch_idx)
            
            # 基于种子生成该patch的伪随机密钥比特
            patch_key_bits = self._generate_patch_key_bits(patch_seed, bits_per_patch)
            
            # 填入对应位置
            start_idx = patch_idx * bits_per_patch
            end_idx = start_idx + bits_per_patch
            K_sem[start_idx:end_idx] = patch_key_bits
        
        # 处理余数位（如果有）
        remaining_bits = total_bits % num_patches
        if remaining_bits > 0:
            # 使用最后一个patch的种子生成余数位
            last_semantic_vector = semantic_vectors[-1]
            last_seed = self._compute_enhanced_patch_seed(last_semantic_vector, num_patches)
            remaining_key_bits = self._generate_patch_key_bits(last_seed, remaining_bits)
            K_sem[-remaining_bits:] = remaining_key_bits
        
        # ⭐ 新增：分布校正机制 ⭐
        K_sem = self._apply_semantic_distribution_correction(K_sem, semantic_vectors)
        
        # 验证伪随机性
        ones_ratio = (K_sem == 1).sum() / total_bits
        print(f"    📊 语义密钥统计: 1的比例 {ones_ratio:.6f} (校正后，严格0.5)")
        print(f"    🔐 每个patch独立绑定: 语义内容 → 确定性种子 → 伪随机密钥")
        
        return K_sem
    
    def _apply_semantic_distribution_correction(self, K_sem: np.ndarray, semantic_vectors: List[torch.Tensor]) -> np.ndarray:
        """
        应用语义分布校正机制，确保K_sem严格满足0.5分布
        
        策略：
        1. 计算当前1的数量和目标数量的差值
        2. 基于语义向量特征生成确定性的位置序列
        3. 按序列翻转对应位置的比特，直到达到严格0.5分布
        
        Args:
            K_sem: 原始语义密钥
            semantic_vectors: 语义向量列表（用于生成确定性校正序列）
            
        Returns:
            校正后的语义密钥（严格0.5分布）
        """
        total_bits = len(K_sem)
        target_ones = total_bits // 2  # 严格的一半
        current_ones = (K_sem == 1).sum()
        
        print(f"    🔧 分布校正: 当前1的数量={current_ones}, 目标={target_ones}")
        
        if current_ones == target_ones:
            print(f"    ✅ 已是完美分布，无需校正")
            return K_sem
        
        # 创建副本进行校正
        K_sem_corrected = K_sem.copy()
        
        # 基于语义向量生成确定性的校正位置序列
        semantic_fingerprint = self._compute_semantic_fingerprint(semantic_vectors)
        correction_seed = int(hashlib.md5(f"{semantic_fingerprint}_correction".encode()).hexdigest()[:8], 16)
        correction_seed = correction_seed & 0x7FFFFFFF
        
        rng = np.random.RandomState(correction_seed)
        position_sequence = rng.permutation(total_bits)
        
        if current_ones > target_ones:
            # 需要将一些1改为0
            excess_ones = current_ones - target_ones
            ones_positions = np.where(K_sem_corrected == 1)[0]
            # 按确定性序列选择要翻转的位置
            positions_to_flip = []
            for pos in position_sequence:
                if pos in ones_positions and len(positions_to_flip) < excess_ones:
                    positions_to_flip.append(pos)
            K_sem_corrected[positions_to_flip] = 0
            print(f"    🔄 翻转{len(positions_to_flip)}个1→0")
            
        elif current_ones < target_ones:
            # 需要将一些0改为1
            deficit_ones = target_ones - current_ones
            zeros_positions = np.where(K_sem_corrected == 0)[0]
            # 按确定性序列选择要翻转的位置
            positions_to_flip = []
            for pos in position_sequence:
                if pos in zeros_positions and len(positions_to_flip) < deficit_ones:
                    positions_to_flip.append(pos)
            K_sem_corrected[positions_to_flip] = 1
            print(f"    🔄 翻转{len(positions_to_flip)}个0→1")
        
        # 验证校正结果
        final_ones = (K_sem_corrected == 1).sum()
        final_ratio = final_ones / total_bits
        print(f"    ✅ 校正完成: 1的数量={final_ones}, 比例={final_ratio:.6f}")
        
        return K_sem_corrected
    
    def _apply_final_distribution_correction_semantic(self, W_loc_final: np.ndarray, semantic_vectors: List[torch.Tensor]) -> np.ndarray:
        """
        应用基于语义的最终分布校正，确保XOR后的结果严格满足0.5分布
        
        Args:
            W_loc_final: XOR后的原始最终水印
            semantic_vectors: 语义向量列表（用于生成确定性校正序列）
            
        Returns:
            校正后的最终水印（严格0.5分布）
        """
        total_bits = len(W_loc_final)
        target_ones = total_bits // 2  # 严格的一半
        current_ones = (W_loc_final == 1).sum()
        
        print(f"    🔧 最终校正: 当前1的数量={current_ones}, 目标={target_ones}")
        
        if current_ones == target_ones:
            print(f"    ✅ 已是完美分布，无需校正")
            return W_loc_final
        
        # 创建副本进行校正
        W_loc_corrected = W_loc_final.copy()
        
        # 基于语义向量生成确定性的校正位置序列
        semantic_fingerprint = self._compute_semantic_fingerprint(semantic_vectors)
        correction_seed = int(hashlib.md5(f"{semantic_fingerprint}_final_correction".encode()).hexdigest()[:8], 16)
        correction_seed = correction_seed & 0x7FFFFFFF
        
        rng = np.random.RandomState(correction_seed)
        position_sequence = rng.permutation(total_bits)
        
        if current_ones > target_ones:
            # 需要将一些1改为0
            excess_ones = current_ones - target_ones
            ones_positions = np.where(W_loc_corrected == 1)[0]
            # 按确定性序列选择要翻转的位置
            positions_to_flip = []
            for pos in position_sequence:
                if pos in ones_positions and len(positions_to_flip) < excess_ones:
                    positions_to_flip.append(pos)
            W_loc_corrected[positions_to_flip] = 0
            print(f"    🔄 翻转{len(positions_to_flip)}个1→0")
            
        elif current_ones < target_ones:
            # 需要将一些0改为1
            deficit_ones = target_ones - current_ones
            zeros_positions = np.where(W_loc_corrected == 0)[0]
            # 按确定性序列选择要翻转的位置
            positions_to_flip = []
            for pos in position_sequence:
                if pos in zeros_positions and len(positions_to_flip) < deficit_ones:
                    positions_to_flip.append(pos)
            W_loc_corrected[positions_to_flip] = 1
            print(f"    🔄 翻转{len(positions_to_flip)}个0→1")
        
        # 验证校正结果
        final_ones = (W_loc_corrected == 1).sum()
        final_ratio = final_ones / total_bits
        print(f"    ✅ 最终校正完成: 1的数量={final_ones}, 比例={final_ratio:.6f}")
        
        return W_loc_corrected
    
    def _compute_semantic_fingerprint(self, semantic_vectors: List[torch.Tensor]) -> str:
        """
        计算语义向量的指纹，用于生成确定性的校正种子
        
        Args:
            semantic_vectors: 语义向量列表
            
        Returns:
            语义指纹字符串
        """
        # 计算所有语义向量的统计特征
        all_vectors = torch.stack(semantic_vectors)  # (64, 768)
        
        # 提取关键统计特征
        mean_val = float(all_vectors.mean())
        std_val = float(all_vectors.std())
        max_val = float(all_vectors.max())
        min_val = float(all_vectors.min())
        
        # 生成指纹
        fingerprint = f"{mean_val:.6f}_{std_val:.6f}_{max_val:.6f}_{min_val:.6f}"
        
        return fingerprint
    
    def _compute_enhanced_patch_seed(self, semantic_vector: torch.Tensor, patch_idx: int) -> int:
        """
        计算增强的patch种子
        
        结合语义向量、patch位置和全局salt生成高质量的种子
        
        Args:
            semantic_vector: 该patch的语义向量 (768维)
            patch_idx: patch索引 (0-63)
            
        Returns:
            增强的种子值
        """
        # 使用优化的SimHash
        from simhash_utils import simhash_single_patch
        
        base_hash = simhash_single_patch(
            semantic_vector,
            num_bits=self.simhash_bits,
            seed=42 + patch_idx
        )
        
        # 多层哈希增强
        # 第一层：结合patch索引
        layer1_input = f"{base_hash}_{patch_idx}_{self.encrypt_random_seed}"
        layer1_hash = int(hashlib.md5(layer1_input.encode()).hexdigest()[:8], 16)
        
        # 第二层：结合语义向量的统计特征
        semantic_stats = {
            'mean': float(semantic_vector.mean()),
            'std': float(semantic_vector.std()),
            'max': float(semantic_vector.max()),
            'min': float(semantic_vector.min())
        }
        layer2_input = f"{layer1_hash}_{semantic_stats['mean']:.6f}_{semantic_stats['std']:.6f}"
        layer2_hash = int(hashlib.sha256(layer2_input.encode()).hexdigest()[:8], 16)
        
        # 最终种子
        final_seed = layer2_hash & 0x7FFFFFFF  # 确保正数
        
        return final_seed
    
    def _generate_patch_key_bits(self, seed: int, length: int) -> np.ndarray:
        """
        基于种子生成高质量的伪随机密钥比特
        
        Args:
            seed: 确定性种子
            length: 需要生成的比特数量
            
        Returns:
            伪随机比特数组 (0/1)
        """
        # 使用numpy的Mersenne Twister生成器，确保高质量伪随机性
        rng = np.random.RandomState(seed)
        key_bits = rng.randint(0, 2, size=length, dtype=np.int32)
        
        return key_bits
    
    def _compute_patch_simhash(self, semantic_vector: torch.Tensor, patch_idx: int) -> int:
        """
        为单个patch计算SimHash值
        
        Args:
            semantic_vector: 该patch的语义向量 (768维)
            patch_idx: patch索引 (0-63)
            
        Returns:
            整数哈希值，用作PRNG种子
        """
        # 导入本地的simhash函数
        from simhash_utils import simhash_single_patch
        
        # 使用优化的单patch SimHash函数
        base_hash = simhash_single_patch(
            semantic_vector, 
            num_bits=self.simhash_bits,
            seed=42 + patch_idx  # 为每个patch使用不同种子
        )
        
        # 结合patch索引和全局salt生成最终种子
        # 复用TAG-WM的确定性随机数生成机制
        combined_seed = f"{base_hash}_{patch_idx}_{self.encrypt_random_seed}"
        
        # 使用哈希函数生成最终的整数种子
        final_seed = int(hashlib.md5(combined_seed.encode()).hexdigest()[:8], 16)
        
        return final_seed
    
    def _generate_patch_watermark_bits(self, seed: int, length: int) -> np.ndarray:
        """
        基于种子生成确定性的水印比特流
        
        Args:
            seed: SimHash生成的种子
            length: 需要生成的比特数量
            
        Returns:
            二进制比特数组 (0/1)
        """
        # 使用numpy的随机数生成器，确保确定性
        rng = np.random.RandomState(seed)
        watermark_bits = rng.randint(0, 2, size=length, dtype=np.uint8)
        return watermark_bits
    
    def save_semantic_map(self, semantic_vectors: List[torch.Tensor], identifier: str):
        """
        保存基准真实语义地图到本地文件
        
        Args:
            semantic_vectors: 64个语义向量
            identifier: 唯一标识符（如prompt的hash）
        """
        # 将语义向量列表转换为张量
        semantic_map = torch.stack(semantic_vectors)  # Shape: (64, 768)
        
        # 保存到指定路径
        save_path = os.path.join(self.semantic_maps_dir, f"semantic_map_{identifier}.pt")
        torch.save(semantic_map, save_path)
        
        print(f"Semantic map saved to: {save_path}")
        return save_path
    
    def load_semantic_map(self, identifier: str) -> Optional[List[torch.Tensor]]:
        """
        从本地文件加载基准语义地图
        
        Args:
            identifier: 唯一标识符
            
        Returns:
            语义向量列表，如果文件不存在则返回None
        """
        load_path = os.path.join(self.semantic_maps_dir, f"semantic_map_{identifier}.pt")
        
        if not os.path.exists(load_path):
            return None
        
        try:
            semantic_map = torch.load(load_path, map_location=self.device)  # Shape: (64, 768)
            semantic_vectors = [semantic_map[i] for i in range(semantic_map.shape[0])]
            print(f"Semantic map loaded from: {load_path}")
            return semantic_vectors
        except Exception as e:
            print(f"Error loading semantic map: {e}")
            return None
    
    def text_to_bits(self, text):
        """将文本转换为二进制比特序列"""
        # 将文本转换为字节，然后转换为比特
        text_bytes = text.encode('utf-8')
        bits = []
        for byte in text_bytes:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)
        return torch.tensor(bits, dtype=torch.float32, device=self.device)
    
    def embedding_seal_loc(self, prompt: str, wm: torch.Tensor, latent_size: Tuple[int, int, int], pipe=None) -> torch.Tensor:
        """
        SEAL-LOC完整嵌入流程
        
        集成六个阶段：
        1. 代理生成与初始表示
        2. 逐补丁语义特征提取  
        3. 动态语义定位水印生成
        4. 版权水印生成（复用TAG-WM）
        5. 双水印联合采样（DMJS）
        6. 扩散生成与解码
        
        Args:
            prompt: 用户文本提示词
            wm: 版权水印比特串 (256位)
            latent_size: 潜空间尺寸 (4, 64, 64)
            pipe: 扩散模型管线 (可选)
            
        Returns:
            latent_noise: 水印化的初始噪声
        """
        print("=== SEAL-LOC Watermark Embedding Pipeline ===")
        
        # 阶段一：生成代理图像
        print("Stage 1: Proxy image generation...")
        if pipe is not None:
            proxy_image = self.generate_proxy_image(prompt, pipe)
        else:
            proxy_image = self.generate_proxy_image(prompt)
        
        # 阶段二：逐补丁语义提取
        print("Stage 2: Patch-wise semantic extraction...")
        semantic_vectors = self.extract_patch_semantics(proxy_image, latent_size)
        
        # 保存基准语义地图
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16]
        self.save_semantic_map(semantic_vectors, prompt_hash)
        
        # 阶段三：XOR基础模式的语义定位水印生成
        print("Stage 3: XOR-based semantic watermark generation...")
        w_loc_s = self.generate_xor_based_semantic_watermark(semantic_vectors, latent_size)
        
        # 阶段四：版权水印生成（复用TAG-WM逻辑）
        print("Stage 4: Copyright watermark generation...")
        latent_len = np.prod(latent_size)
        tlt_dummy = torch.zeros(latent_len, device=self.device)  # 占位符，将被w_loc_s替代
        w_cop = self._generate_copyright_watermark(wm, latent_len)
        
        # 阶段五：双水印联合采样（DMJS）
        print("Stage 5: Dual watermark joint sampling (DMJS)...")
        latent_noise, wm_repeat = self.embedding_wm_tlt(w_cop, w_loc_s, latent_size)
        
        print("=== SEAL-LOC Embedding Completed ===")
        return latent_noise
    
    def _generate_copyright_watermark(self, wm: torch.Tensor, latent_len: int) -> torch.Tensor:
        """
        生成版权水印（复用TAG-WM的逻辑）
        
        Args:
            wm: 原始水印比特串
            latent_len: 潜空间总长度
            
        Returns:
            扩展并加密后的版权水印
        """
        # 重复展开水印以覆盖潜空间
        wm_len = wm.shape[0]
        wm_times = int(latent_len // wm_len)
        remain_wm_len = latent_len % wm_len
        wm_repeat = torch.concat([wm.repeat(wm_times), wm[:remain_wm_len]], dim=0)
        
        # 流加密（可选）- 确保输入是整数类型
        wm_repeat_int = wm_repeat.cpu().numpy().astype(np.uint8)
        wm_repeat_encrypt = self.stream_key_encrypt(wm_repeat_int)
        
        return torch.from_numpy(wm_repeat_encrypt).float().to(self.device)


if __name__ == "__main__":
    # 简单测试
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化SEAL-LOC嵌入器
    seal_loc = SEALLOCEmbedder(device=device)
    
    print("SEAL-LOC Embedder initialized successfully!")
    print(f"Number of patches: {seal_loc.num_patches}")
    print(f"Semantic vector dimension: {seal_loc.semantic_vector_dim}") 