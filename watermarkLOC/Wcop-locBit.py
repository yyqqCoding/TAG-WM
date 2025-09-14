import os
import sys
import argparse
import hashlib
from typing import Dict, Any

import torch
import numpy as np
from PIL import Image
from diffusers.schedulers import DDIMScheduler

# 环境设置
os.environ.setdefault('DISABLE_DVRD', '1')

# 加入仓库根目录以使用包路径导入
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
	sys.path.append(repo_root)

# 导入可反演Stable Diffusion管线
try:
	from applied_to_sd2.inverse_stable_diffusion import InversableStableDiffusionPipeline
except Exception as e:
	print(f"[ERROR] 需要可反演管线 InversableStableDiffusionPipeline，但导入失败: {e}")
	raise

# 导入TAG-WM嵌入器（用于解码）
try:
	from applied_to_sd2.watermark_embedder import WatermarkEmbedder
except Exception:
	from watermark_embedder import WatermarkEmbedder

# 本地模型路径解析
try:
	from model_config import get_model_path
except Exception:
	def get_model_path(model_name: str) -> str:
		return model_name


def md5_16(s: str) -> str:
	return hashlib.md5(s.encode()).hexdigest()[:16]


def preprocess_pil_to_tensor(image: Image.Image, height: int, width: int, dtype: torch.dtype, device: str) -> torch.Tensor:
	resample = getattr(Image, 'Resampling', Image)
	image_resized = image.resize((width, height), resample=getattr(resample, 'LANCZOS', Image.BICUBIC))
	image_np = np.array(image_resized).astype(np.float32) / 255.0
	image_np = image_np[None].transpose(0, 3, 1, 2)  # (1,3,H,W)
	image_tensor = torch.from_numpy(image_np).to(device, dtype=dtype)
	image_tensor = 2.0 * image_tensor - 1.0  # 归一化到[-1,1]
	return image_tensor


def load_pipe(model_id: str, device: str) -> InversableStableDiffusionPipeline:
	model_path = get_model_path(model_id)
	print(f"[Load] 使用本地模型路径: {model_path}")
	pipe = InversableStableDiffusionPipeline.from_pretrained(
		model_path,
		torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
		local_files_only=True,
	)
	if hasattr(pipe, 'safety_checker'):
		pipe.safety_checker = None
	pipe = pipe.to(device)
	# 统一反演调度器为DDIM
	try:
		pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
	except Exception as _e:
		print(f"[Load] 切换DDIM调度器失败: {_e}")
	return pipe


@torch.inference_mode()
def ddim_invert_to_noise(pipe: InversableStableDiffusionPipeline, image_tensor: torch.Tensor, steps: int, guidance_scale: float = 1.0, prompt: str = '', blind: bool = True) -> torch.Tensor:
	# 盲提取：不知prompt时使用uncond嵌入并关闭CFG
	use_blind = blind or (prompt is None) or (str(prompt).strip() == '')
	if use_blind:
		text_embeddings = pipe.get_text_embedding("")
		guidance_scale = 1.0
	else:
		if guidance_scale is not None and guidance_scale > 1.0:
			uncond = pipe.get_text_embedding("")
			cond = pipe.get_text_embedding(prompt)
			text_embeddings = torch.cat([uncond, cond], dim=0)
		else:
			text_embeddings = pipe.get_text_embedding(prompt)
	# VAE编码图像到潜空间（不采样以减少随机性）
	latents_0 = pipe.get_image_latents(image_tensor, sample=False)
	# DDIM正向（到噪声方向）
	latents_T = pipe.forward_diffusion(
		text_embeddings=text_embeddings,
		latents=latents_0,
		num_inference_steps=steps,
		guidance_scale=guidance_scale if guidance_scale is not None else 1.0,
	)
	return latents_T


def compute_bit_accuracy(a: np.ndarray, b: np.ndarray) -> float:
	assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
	return float((a == b).sum()) / float(a.size)


def main():
	parser = argparse.ArgumentParser(description='Wcop-locBit: 反演并评估W_cop与W_loc^S比特准确率')
	parser.add_argument('--prompt', type=str, default='')
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
	parser.add_argument('--height', type=int, default=512)
	parser.add_argument('--width', type=int, default=512)
	parser.add_argument('--num_inference_steps', type=int, default=50)
	parser.add_argument('--guidance_scale', type=float, default=1.0)
	parser.add_argument('--wm_len', type=int, default=256)
	parser.add_argument('--tlt_intervals_num', type=int, default=3)
	parser.add_argument('--image_path', type=str, default='')
	parser.add_argument('--w_cop_path', type=str, default='')
	parser.add_argument('--w_loc_path', type=str, default='')
	parser.add_argument('--z_tw_path', type=str, default='')
	parser.add_argument('--blind', action='store_true', default=True)
	args = parser.parse_args()

	device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
	ph = md5_16(args.prompt) if args.prompt.strip() != '' else ''

	# 推断默认路径
	base_dir = os.path.dirname(__file__)
	default_image_path = os.path.join(base_dir, 'output', 'stage6', f'image_{ph}.png') if ph else ''
	default_wcop_path = os.path.join(base_dir, 'output', 'stage5', f'w_cop_{ph}.pt') if ph else ''
	default_wloc_path = os.path.join(base_dir, 'output', 'stage4', f'w_loc_s_{ph}.pt') if ph else ''
	default_ztw_path = os.path.join(base_dir, 'output', 'stage6', f'z_tw_{ph}.pt') if ph else ''

	image_path = args.image_path or default_image_path
	w_cop_path = args.w_cop_path or default_wcop_path
	w_loc_path = args.w_loc_path or default_wloc_path
	z_tw_path = args.z_tw_path  # 可选，仅用于L2对比

	# 盲提取时若未提供原始Wcop/Wloc路径，无法做准确率对比
	if args.blind and (not w_cop_path or not w_loc_path):
		raise ValueError("盲提取模式下无法根据prompt推断文件名，请通过 --w_cop_path 与 --w_loc_path 显式提供嵌入期保存的Wcop与Wloc路径")

	# 加载管线（本地）
	pipe = load_pipe(args.model_id, device)

	# 读取带水印图片
	assert os.path.exists(image_path), f"水印图不存在: {image_path}"
	watermarked_img = Image.open(image_path).convert('RGB')
	img_tensor = preprocess_pil_to_tensor(watermarked_img, args.height, args.width, pipe.text_encoder.dtype, device)

	# DDIM反演至噪声
	latents_T = ddim_invert_to_noise(pipe, img_tensor, args.num_inference_steps, args.guidance_scale, args.prompt, args.blind)
	print(f"[Invert] 反演得到噪声形状: {tuple(latents_T.shape)}")

	# 反解Wcop与Wloc
	embedder = WatermarkEmbedder(
		wm_len=args.wm_len,
		center_interval_ratio=0.5,
		shuffle_random_seed=133563,
		encrypt_random_seed=133563,
		tlt_intervals_num=args.tlt_intervals_num,
		device=device,
	)
	wm_repeat, reversed_tlt = embedder.deembedding_wm_tlt(latents_T)
	reversed_wm_bits = embedder.calc_watermark(args.wm_len, wm_repeat, with_tamper_loc=False)

	# 加载嵌入期保存的Wcop/Wloc
	assert os.path.exists(w_cop_path), f"W_cop文件不存在: {w_cop_path}"
	assert os.path.exists(w_loc_path), f"W_loc^S文件不存在: {w_loc_path}"
	orig_wcop = torch.load(w_cop_path, map_location='cpu')  # (wm_len,)
	orig_wloc = torch.load(w_loc_path, map_location='cpu')  # (C,H,W)

	# 计算bit准确率
	rev_wm_np = reversed_wm_bits.detach().cpu().numpy().astype(np.uint8)
	orig_wm_np = orig_wcop.detach().cpu().numpy().astype(np.uint8)
	acc_wcop = compute_bit_accuracy(rev_wm_np, orig_wm_np)

	rev_wloc_np = reversed_tlt.astype(np.uint8).reshape(-1)
	orig_wloc_np = orig_wloc.detach().cpu().numpy().astype(np.uint8).reshape(-1)
	acc_wloc = compute_bit_accuracy(rev_wloc_np, orig_wloc_np)

	print(f"W_cop bit accuracy: {acc_wcop:.6f}")
	print(f"W_loc^S bit accuracy: {acc_wloc:.6f}")

	# 可选：对比反演噪声与嵌入期保存的Z_T^w的L2
	if z_tw_path and os.path.exists(z_tw_path):
		saved_ztw = torch.load(z_tw_path, map_location=device)
		l2 = torch.norm(saved_ztw - latents_T).item()
		print(f"Z_T^w inversion L2 distance: {l2:.6f}")


if __name__ == '__main__':
	main() 