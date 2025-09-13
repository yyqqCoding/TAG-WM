"""
SEAL-LOC 完整测试系统

完整流程测试：
1. 使用TAG-WM生成版权水印Wcop
2. 使用创新的语义定位水印生成Wloc
3. 使用原有的DMJS生成初始噪声，利用扩散模型生成图片
4. 采用原有的噪声重建方案得到版权水印和定位水印，计算bit精度

参考TAG-WM原始代码，直接复用扩散模型加载和提示词处理
"""

import torch
import numpy as np
import os
import sys
import hashlib
import argparse
import time
from PIL import Image
from typing import Tuple, List, Optional
import math
import zlib
import random

# 设置环境变量禁用DVRD
os.environ['DISABLE_DVRD'] = '1'

# 添加路径
sys.path.append('../applied_to_sd2')
sys.path.append('../baseline')

# 导入TAG-WM核心组件
from watermark_embedder import WatermarkEmbedder

# 导入扩散模型组件
try:
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from modified_stable_diffusion import ModifiedStableDiffusionPipeline
except ImportError:
    try:
        from diffusers import StableDiffusionPipeline as InversableStableDiffusionPipeline
        print("Warning: Using standard diffusers instead of modified versions")
    except ImportError:
        print("Error: Could not import diffusers. Please install with: pip install diffusers")
        raise

# 导入SEAL-LOC组件
try:
    from seal_loc_embedder import SEALLOCEmbedder
    from simhash_utils import simhash_single_patch
    from patch_utils import calculate_patch_l2, transform_img
except ImportError as e:
    print(f"Warning: Could not import SEAL-LOC components: {e}")
    print("Please make sure you are running from the watermarkLOC directory")
    raise

# 导入baseline工具函数
sys.path.append('../baseline')
try:
    from caption_pairs import compute_simhash, generate_caption
except ImportError:
    # 如果导入失败，提供备用实现
    print("Warning: Could not import from caption_pairs, using fallback implementations")
    
    def compute_simhash(embedding, num_patches, num_bits, seed):
        """备用的compute_simhash实现"""
        import random
        import zlib
        import torch
        
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
    
    def generate_caption(image, processor=None, model=None, device='cuda'):
        """备用的generate_caption实现"""
        return "A generated image"  # 简单的备用实现


def create_baseline_simhash_function():
    """创建与baseline兼容的simhash函数"""
    def simhash(embedding, k, b, seed):
        """
        兼容baseline的simhash函数
        参数：
        - embedding: 语义嵌入向量
        - k: patch数量 (64)
        - b: simhash比特数 (7)
        - seed: 随机种子
        """
        return compute_simhash(embedding, k, b, seed)
    
    return simhash


class CompleteSEALLOCTest:
    """完整的SEAL-LOC测试系统"""
    
    def __init__(self, device='cuda', model_id='stabilityai/stable-diffusion-2-1-base'):
        self.device = device
        self.model_id = model_id
        
        # 本地模型路径映射
        self.local_model_paths = {
            'stabilityai/stable-diffusion-2-1-base': '/home/wang003/.cache/modelscope/hub/models/AI-ModelScope/stable-diffusion-2-1-base',
            'laion/CLIP-ViT-g-14-laion2B-s12B-b42K': '/media/wang003/liyongqing/difusion/cache/models--laion--CLIP-ViT-g-14-laion2B-s12B-b42K/snapshots/4b0305adc6802b2632e11cbe6606a9bdd43d35c9',
            'Salesforce/blip2-flan-t5-xl': '/home/wang003/.cache/modelscope/hub/models/zimuwangnlp/flan-t5-xl',
            'kasraarabi/finetuned-caption-embedding': '/home/wang003/.cache/modelscope/hub/models/kasraarabi/finetuned-caption-embedding'
        }
        
        print(f"🚀 初始化SEAL-LOC完整测试系统 (设备: {device})")
        print(f"📝 扩散模型: {model_id}")
        
        # 初始化SEAL-LOC嵌入器
        self.seal_loc_embedder = SEALLOCEmbedder(device=device)
        
        # 初始化TAG-WM水印嵌入器（用于版权水印）
        self.tag_wm_embedder = WatermarkEmbedder(
            wm_len=256,
            center_interval_ratio=0.5,
            shuffle_random_seed=133563,
            encrypt_random_seed=133563,
            tlt_intervals_num=3,
            device=device
        )
        
        # 扩散模型管线
        self.pipe = None
        
        # 创建baseline兼容的simhash函数
        self.simhash = create_baseline_simhash_function()
        
        print("✅ SEAL-LOC测试系统初始化完成")
    
    def load_diffusion_model(self):
        """加载扩散模型，复用TAG-WM原始代码逻辑"""
        print("🔄 加载Stable Diffusion 2.1模型...")
        
        # 尝试使用本地路径
        model_path = self.local_model_paths.get(self.model_id, self.model_id)
        
        try:
            print(f"📁 尝试从本地路径加载: {model_path}")
            # 使用可反演的Stable Diffusion管线
            self.pipe = InversableStableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                local_files_only=True,  # 强制使用本地文件
            )
            self.pipe.safety_checker = None  # 禁用安全检查器
            self.pipe = self.pipe.to(self.device)
            print("✅ 扩散模型加载成功")
            
        except Exception as e:
            print(f"❌ 本地模型加载失败: {e}")
            print("🔄 尝试在线加载...")
            try:
                self.pipe = InversableStableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    revision='fp16',
                )
                self.pipe.safety_checker = None
                self.pipe = self.pipe.to(self.device)
                print("✅ 在线模型加载成功")
            except Exception as e2:
                print(f"❌ 在线模型加载也失败: {e2}")
                raise e2
    
    def generate_copyright_watermark(self, message: str = "SEAL-LOC-TEST") -> torch.Tensor:
        """生成版权水印Wcop，使用TAG-WM方案"""
        print("🔒 生成版权水印 (Wcop)...")
        
        # 将消息转换为二进制
        message_bytes = message.encode('utf-8')
        message_bits = ''.join(format(byte, '08b') for byte in message_bytes)
        
        # 截断或填充到256位
        if len(message_bits) > 256:
            message_bits = message_bits[:256]
        else:
            message_bits = message_bits.ljust(256, '0')
        
        # 转换为tensor
        wm_tensor = torch.tensor([int(bit) for bit in message_bits], 
                                dtype=torch.float32, device=self.device)
        
        print(f"✅ 版权水印生成完成 (长度: {len(wm_tensor)})")
        return wm_tensor
    
    def generate_semantic_location_watermark(self, prompt: str, latent_size: Tuple[int, int, int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """生成语义定位水印Wloc，使用SEAL-LOC创新方案"""
        print("🎯 生成语义定位水印 (Wloc)...")
        
        try:
            # 使用SEAL-LOC嵌入器生成语义定位水印
            # 先生成代理图像
            print("  📸 生成代理图像...")
            proxy_image = self.pipe(prompt).images[0]
            
            # 提取逐patch语义特征
            print("  🧠 提取patch语义特征...")
            semantic_vectors = self.seal_loc_embedder.extract_patch_semantics(
                proxy_image, latent_size
            )
            
            # 生成动态语义定位水印
            print("  🔗 生成动态语义定位水印...")
            w_loc_s = self.seal_loc_embedder.generate_dynamic_semantic_watermark(
                semantic_vectors, latent_size
            )
            
            print(f"✅ 语义定位水印生成完成 (形状: {w_loc_s.shape})")
            return w_loc_s, semantic_vectors
            
        except Exception as e:
            print(f"❌ 语义定位水印生成失败: {e}")
            # 回退到模拟方案
            print("🔄 回退到模拟语义向量...")
            return self.generate_simulated_location_watermark(latent_size)
    
    def generate_simulated_location_watermark(self, latent_size: Tuple[int, int, int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """生成模拟的语义定位水印（用于测试）"""
        print("🎲 生成模拟语义定位水印...")
        
        # 模拟64个patch的语义向量
        num_patches = 64
        semantic_dim = 768
        
        semantic_vectors = []
        for i in range(num_patches):
            # 生成确定性的模拟语义向量
            torch.manual_seed(42 + i)
            semantic_vector = torch.randn(semantic_dim, device=self.device)
            semantic_vector = semantic_vector / torch.norm(semantic_vector)  # 归一化
            semantic_vectors.append(semantic_vector)
        
        # 生成水印比特
        total_bits = latent_size[0] * latent_size[1] * latent_size[2]
        w_loc_bits = []
        
        for i, semantic_vector in enumerate(semantic_vectors):
            # 使用simhash生成确定性种子
            hash_seed = simhash_single_patch(semantic_vector, num_bits=7, seed=42 + i)
            
            # 计算每个patch需要的比特数
            bits_per_patch = total_bits // num_patches
            
            # 生成确定性水印比特
            np.random.seed(hash_seed & 0xFFFFFFFF)
            patch_bits = np.random.randint(0, 2, size=bits_per_patch)
            w_loc_bits.extend(patch_bits)
        
        # 调整到精确的总比特数
        if len(w_loc_bits) > total_bits:
            w_loc_bits = w_loc_bits[:total_bits]
        elif len(w_loc_bits) < total_bits:
            w_loc_bits.extend([0] * (total_bits - len(w_loc_bits)))
        
        # 转换为tensor并reshape
        w_loc_s = torch.tensor(w_loc_bits, dtype=torch.float32, device=self.device)
        w_loc_s = w_loc_s.reshape(latent_size)
        
        print(f"✅ 模拟语义定位水印生成完成 (形状: {w_loc_s.shape})")
        return w_loc_s, semantic_vectors
    
    def generate_initial_noise_dmjs(self, w_cop: torch.Tensor, w_loc: torch.Tensor) -> torch.Tensor:
        """使用DMJS生成初始噪声，复用TAG-WM方案"""
        print("🎲 使用DMJS生成初始噪声...")
        
        try:
            # 将tensor转换为1维
            w_loc_flat = w_loc.flatten()
            total_len = len(w_loc_flat)
            
            # 将版权水印重复扩展到与定位水印相同的长度
            w_cop_expanded = w_cop.repeat((total_len + len(w_cop) - 1) // len(w_cop))[:total_len]
            
            print(f"  版权水印扩展: {len(w_cop)} → {len(w_cop_expanded)}")
            print(f"  定位水印长度: {len(w_loc_flat)}")
            
            # 使用TAG-WM的DMJS采样
            sampled_noise_flat = self.tag_wm_embedder.denseWMandDenseFixedTLTtruncSampling(w_cop_expanded, w_loc_flat)
            
            # 打乱噪声（TAG-WM的标准流程）
            sampled_noise_flat = self.tag_wm_embedder.shuffle(sampled_noise_flat)
            
            # 重塑为latent形状并添加batch维度
            latent_noise = sampled_noise_flat.reshape(1, *w_loc.shape)
            
            print("✅ 使用TAG-WM的DMJS采样成功")
            
        except Exception as e:
            print(f"Warning: DMJS failed, using fallback: {e}")
            # 备用方案：简单的正态分布噪声
            latent_noise = torch.randn(1, *w_loc.shape, device=self.device, dtype=torch.float16)
        
        print(f"✅ 初始噪声生成完成 (形状: {latent_noise.shape})")
        return latent_noise
    
    def generate_watermarked_image(self, prompt: str, latent_noise: torch.Tensor) -> Image.Image:
        """使用水印化噪声生成图像"""
        print("🎨 生成水印化图像...")
        
        # 使用水印化噪声作为初始latents
        image = self.pipe(prompt, latents=latent_noise).images[0]
        
        print("✅ 水印化图像生成完成")
        return image
    
    def reconstruct_watermarks(self, image: Image.Image, original_w_cop: torch.Tensor, 
                              original_w_loc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """重建水印并计算精度"""
        print("🔍 重建水印...")
        
        # 将图像转换为tensor
        image_tensor = transform_img(image).unsqueeze(0).to(self.device)
        image_tensor = image_tensor.to(dtype=self.pipe.vae.dtype)
        
        # 获取图像的latent表示
        image_latents = self.pipe.get_image_latents(image_tensor, sample=False)
        
        # DDIM反转得到初始噪声
        print("  🔄 执行DDIM反转...")
        reconstructed_noise = self.pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=self.pipe.get_text_embedding(''),
            guidance_scale=1,
            num_inference_steps=50,
        )
        
        # 使用TAG-WM重建水印
        print("  📊 重建版权水印和定位水印...")
        reconstructed_w_cop, reconstructed_w_loc = self.tag_wm_embedder.extract_watermark(
            reconstructed_noise
        )
        
        # 计算精度
        metrics = self.calculate_bit_accuracy(
            original_w_cop, original_w_loc,
            reconstructed_w_cop, reconstructed_w_loc
        )
        
        print("✅ 水印重建完成")
        return reconstructed_w_cop, reconstructed_w_loc, metrics
    
    def calculate_bit_accuracy(self, orig_w_cop: torch.Tensor, orig_w_loc: torch.Tensor,
                              recon_w_cop: torch.Tensor, recon_w_loc: torch.Tensor) -> dict:
        """计算水印比特精度"""
        print("📏 计算比特精度...")
        
        # 版权水印精度
        cop_accuracy = (orig_w_cop == recon_w_cop).float().mean().item()
        
        # 定位水印精度
        loc_accuracy = (orig_w_loc == recon_w_loc).float().mean().item()
        
        # 计算L2距离（用于SEAL方案评估）
        l2_distance = calculate_patch_l2(
            orig_w_loc.unsqueeze(0), 
            recon_w_loc.unsqueeze(0), 
            k=64
        )
        
        metrics = {
            'copyright_accuracy': cop_accuracy,
            'location_accuracy': loc_accuracy,
            'l2_distance': l2_distance,
            'total_bits_cop': len(orig_w_cop),
            'total_bits_loc': orig_w_loc.numel(),
            'correct_bits_cop': int((orig_w_cop == recon_w_cop).sum()),
            'correct_bits_loc': int((orig_w_loc == recon_w_loc).sum())
        }
        
        print(f"📊 版权水印精度: {cop_accuracy:.4f}")
        print(f"📊 定位水印精度: {loc_accuracy:.4f}")
        print(f"📊 L2距离: {l2_distance:.4f}")
        
        return metrics
    
    def run_complete_test(self, prompt: str = "A beautiful landscape with mountains and trees", 
                         output_dir: str = "output/complete_test") -> dict:
        """运行完整测试流程"""
        print("🚀 开始SEAL-LOC完整测试流程")
        print(f"📝 测试提示词: {prompt}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        try:
            # 1. 加载扩散模型
            if self.pipe is None:
                self.load_diffusion_model()
            
            # 2. 生成版权水印
            w_cop = self.generate_copyright_watermark()
            
            # 3. 生成语义定位水印
            latent_size = (4, 64, 64)
            w_loc, semantic_vectors = self.generate_semantic_location_watermark(prompt, latent_size)
            
            # 4. 使用DMJS生成初始噪声
            latent_noise = self.generate_initial_noise_dmjs(w_cop, w_loc)
            
            # 5. 生成水印化图像
            watermarked_image = self.generate_watermarked_image(prompt, latent_noise)
            
            # 6. 重建水印并计算精度
            recon_w_cop, recon_w_loc, metrics = self.reconstruct_watermarks(
                watermarked_image, w_cop, w_loc
            )
            
            # 7. 保存结果
            self.save_test_results(
                output_dir, prompt, watermarked_image, 
                w_cop, w_loc, recon_w_cop, recon_w_loc, 
                semantic_vectors, metrics
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"✅ 完整测试流程完成！耗时: {total_time:.2f}秒")
            
            # 添加时间信息到metrics
            metrics['total_time'] = total_time
            
            return metrics
            
        except Exception as e:
            print(f"❌ 测试流程失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def save_test_results(self, output_dir: str, prompt: str, image: Image.Image,
                         w_cop: torch.Tensor, w_loc: torch.Tensor,
                         recon_w_cop: torch.Tensor, recon_w_loc: torch.Tensor,
                         semantic_vectors: List[torch.Tensor], metrics: dict):
        """保存测试结果"""
        print("💾 保存测试结果...")
        
        # 保存图像
        image_path = os.path.join(output_dir, "watermarked_image.png")
        image.save(image_path)
        
        # 保存水印数据
        watermark_data = {
            'prompt': prompt,
            'original_w_cop': w_cop.cpu().numpy(),
            'original_w_loc': w_loc.cpu().numpy(),
            'reconstructed_w_cop': recon_w_cop.cpu().numpy(),
            'reconstructed_w_loc': recon_w_loc.cpu().numpy(),
            'metrics': metrics
        }
        
        watermark_path = os.path.join(output_dir, "watermark_data.npz")
        np.savez(watermark_path, **watermark_data)
        
        # 保存语义向量
        semantic_data = {f'semantic_vector_{i}': vec.cpu().numpy() 
                        for i, vec in enumerate(semantic_vectors)}
        semantic_path = os.path.join(output_dir, "semantic_vectors.npz")
        np.savez(semantic_path, **semantic_data)
        
        # 保存metrics到文本文件
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"SEAL-LOC Complete Test Results\n")
            f.write(f"================================\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Copyright Watermark Accuracy: {metrics['copyright_accuracy']:.4f}\n")
            f.write(f"Location Watermark Accuracy: {metrics['location_accuracy']:.4f}\n")
            f.write(f"L2 Distance: {metrics['l2_distance']:.4f}\n")
            f.write(f"Total Time: {metrics.get('total_time', 0):.2f}s\n")
            f.write(f"Correct Copyright Bits: {metrics['correct_bits_cop']}/{metrics['total_bits_cop']}\n")
            f.write(f"Correct Location Bits: {metrics['correct_bits_loc']}/{metrics['total_bits_loc']}\n")
        
        print(f"✅ 测试结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='SEAL-LOC Complete Test')
    parser.add_argument('--prompt', type=str, 
                       default="A beautiful landscape with mountains and trees",
                       help='测试提示词')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='设备 (cuda/cpu)')
    parser.add_argument('--model_id', type=str, 
                       default='stabilityai/stable-diffusion-2-1-base',
                       help='扩散模型ID')
    parser.add_argument('--output_dir', type=str, 
                       default='output/complete_test',
                       help='输出目录')
    parser.add_argument('--num_tests', type=int, default=1,
                       help='测试次数')
    
    args = parser.parse_args()
    
    # 初始化测试系统
    tester = CompleteSEALLOCTest(device=args.device, model_id=args.model_id)
    
    # 运行测试
    all_metrics = []
    
    for i in range(args.num_tests):
        print(f"\n🔄 运行第 {i+1}/{args.num_tests} 次测试")
        
        test_output_dir = os.path.join(args.output_dir, f"test_{i+1}")
        
        try:
            metrics = tester.run_complete_test(
                prompt=args.prompt,
                output_dir=test_output_dir
            )
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"❌ 第 {i+1} 次测试失败: {e}")
            continue
    
    # 计算平均指标
    if all_metrics:
        avg_metrics = {
            'avg_copyright_accuracy': np.mean([m['copyright_accuracy'] for m in all_metrics]),
            'avg_location_accuracy': np.mean([m['location_accuracy'] for m in all_metrics]),
            'avg_l2_distance': np.mean([m['l2_distance'] for m in all_metrics]),
            'avg_total_time': np.mean([m.get('total_time', 0) for m in all_metrics])
        }
        
        print(f"\n📊 平均测试结果 ({len(all_metrics)} 次测试):")
        print(f"平均版权水印精度: {avg_metrics['avg_copyright_accuracy']:.4f}")
        print(f"平均定位水印精度: {avg_metrics['avg_location_accuracy']:.4f}")
        print(f"平均L2距离: {avg_metrics['avg_l2_distance']:.4f}")
        print(f"平均耗时: {avg_metrics['avg_total_time']:.2f}秒")
        
        # 保存平均结果
        summary_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"SEAL-LOC Complete Test Summary\n")
            f.write(f"==============================\n")
            f.write(f"Number of tests: {len(all_metrics)}\n")
            f.write(f"Average Copyright Accuracy: {avg_metrics['avg_copyright_accuracy']:.4f}\n")
            f.write(f"Average Location Accuracy: {avg_metrics['avg_location_accuracy']:.4f}\n")
            f.write(f"Average L2 Distance: {avg_metrics['avg_l2_distance']:.4f}\n")
            f.write(f"Average Time: {avg_metrics['avg_total_time']:.2f}s\n")
    
    print("🎉 所有测试完成！")


if __name__ == "__main__":
    main() 