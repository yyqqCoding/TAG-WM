import os
import sys
import argparse
import hashlib
import json
from typing import Optional, Dict, Any, Tuple

import torch
import numpy as np
from PIL import Image
from diffusers.schedulers import DDIMScheduler

# 关闭可训练DVRD（检测阶段1无需）
os.environ.setdefault('DISABLE_DVRD', '1')

# 确保可从仓库根目录导入
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
	sys.path.append(REPO_ROOT)

# 复用可反演Stable Diffusion管线与嵌入器
try:
	from applied_to_sd2.inverse_stable_diffusion import InversableStableDiffusionPipeline
except Exception as e:
	print(f"[ERROR] 无法导入 InversableStableDiffusionPipeline: {e}")
	raise

try:
	from applied_to_sd2.watermark_embedder import WatermarkEmbedder
except Exception as e:
	print(f"[ERROR] 无法导入 WatermarkEmbedder: {e}")
	raise

# 模型本地路径解析
try:
	from watermarkLOC.model_config import get_model_path
except Exception:
	def get_model_path(model_name: str) -> str:
		return model_name


def _md5_16(s: str) -> str:
	return hashlib.md5(s.encode()).hexdigest()[:16]


def _preprocess_pil_to_tensor(image: Image.Image, height: int, width: int, dtype: torch.dtype, device: str) -> torch.Tensor:
	resample = getattr(Image, 'Resampling', Image)
	image_resized = image.resize((width, height), resample=getattr(resample, 'LANCZOS', Image.BICUBIC))
	image_np = np.array(image_resized).astype(np.float32) / 255.0
	image_np = image_np[None].transpose(0, 3, 1, 2)  # (1,3,H,W)
	image_tensor = torch.from_numpy(image_np).to(device, dtype=dtype)
	image_tensor = 2.0 * image_tensor - 1.0
	return image_tensor

# 新增：严格解析本地模型路径
def _resolve_local_model_path(model_id: str, sd_local_path: Optional[str]) -> str:
	"""
	解析并校验本地模型目录：
	优先级：
	1) 显式传入的 sd_local_path
	2) watermarkLOC.model_config.get_model_path(model_id)
	3) watermarkLOC.model_config.get_model_path(model_id.split('/')[-1])
	4) 环境变量（SD_2_1_BASE_PATH / STABLE_DIFFUSION_2_1_BASE_PATH / SD_LOCAL_PATH）
	若均无效，则抛出异常。
	"""
	candidates = []
	if sd_local_path and sd_local_path.strip():
		candidates.append(sd_local_path.strip())
	# 通过配置解析两次（完整名与短名）
	try:
		cand = get_model_path(model_id)
		if cand and cand != model_id:
			candidates.append(cand)
	except Exception:
		pass
	short_id = model_id.split('/')[-1] if '/' in model_id else model_id
	try:
		cand = get_model_path(short_id)
		if cand and cand not in candidates:
			candidates.append(cand)
	except Exception:
		pass
	# 环境变量兜底
	for env_key in [
		'SD_2_1_BASE_PATH',
		'STABLE_DIFFUSION_2_1_BASE_PATH',
		'SD_LOCAL_PATH',
	]:
		val = os.getenv(env_key, '').strip()
		if val:
			candidates.append(val)

	# 选择第一个存在的目录
	for p in candidates:
		if os.path.isdir(p):
			return p

	raise FileNotFoundError(
		"未能找到可用的本地 Stable Diffusion 模型目录，请：\n"
		"- 在 watermarkLOC/model_config.py 中配置 get_model_path 返回本地绝对路径，或\n"
		"- 通过 --sd_local_path 显式传入模型目录，或\n"
		"- 设置环境变量 SD_2_1_BASE_PATH=/abs/path/to/stable-diffusion-2-1-base\n"
		f"传入的 model_id='{model_id}', 候选路径为: {candidates}"
	)


def _load_pipe(model_id: str, device: str, sd_local_path: Optional[str] = None) -> InversableStableDiffusionPipeline:
	# 严格仅本地加载
	model_path = _resolve_local_model_path(model_id, sd_local_path)
	print(f"[Stage1] 本地加载模型: {model_id} @ {model_path}")
	pipe = InversableStableDiffusionPipeline.from_pretrained(
		model_path,
		torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
		local_files_only=True,
	)
	if hasattr(pipe, 'safety_checker'):
		pipe.safety_checker = None
	pipe = pipe.to(device)
	# 统一DDIM调度器
	try:
		pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
	except Exception as _e:
		print(f"[Stage1] 切换DDIM失败: {_e}")
	return pipe


def _build_text_embeddings(pipe: InversableStableDiffusionPipeline, prompt: Optional[str], guidance_scale: Optional[float], blind: bool) -> Tuple[torch.Tensor, float]:
	use_blind = blind or (prompt is None) or (str(prompt).strip() == '')
	if use_blind:
		text_embeddings = pipe.get_text_embedding("")
		return text_embeddings, 1.0
	else:
		if guidance_scale is not None and guidance_scale > 1.0:
			uncond = pipe.get_text_embedding("")
			cond = pipe.get_text_embedding(prompt)
			text_embeddings = torch.cat([uncond, cond], dim=0)
			return text_embeddings, float(guidance_scale)
		else:
			text_embeddings = pipe.get_text_embedding(prompt)
			return text_embeddings, 1.0


@torch.inference_mode()
def run_stage1(
	image_path: str,
	*,
	device: str = 'cuda',
	model_id: str = 'stabilityai/stable-diffusion-2-1-base',
	height: int = 512,
	width: int = 512,
	num_inference_steps: int = 50,
	guidance_scale: float = 1.0,
	prompt: Optional[str] = None,
	blind: bool = True,
	wm_len: int = 256,
	tlt_intervals_num: int = 3,
	center_interval_ratio: float = 0.5,
	shuffle_random_seed: int = 133563,
	encrypt_random_seed: int = 133563,
	output_root: Optional[str] = None,
	hash_code: Optional[str] = None,
	sd_local_path: Optional[str] = None,
) -> Dict[str, Any]:
	"""
	阶段一：隐空间重构与水印初步解码
	- 输入：水印图像
	- 输出：Z0^w、Z_T^w、~W_cop（与潜空间同维度）、~W_loc（与潜空间同维度）
	- 完全复用 InversableStableDiffusionPipeline 与 WatermarkEmbedder
	"""
	# 设备
	device = device if torch.cuda.is_available() and str(device).startswith('cuda') else 'cpu'
	# 输出目录
	output_root = output_root or os.path.join(os.path.dirname(__file__), 'output')
	# 推断hash
	if hash_code is None or len(str(hash_code).strip()) == 0:
		basename = os.path.basename(image_path)
		if basename.startswith('image_') and basename.endswith('.png'):
			try:
				hash_code = basename[len('image_'):-len('.png')]
			except Exception:
				hash_code = _md5_16(os.path.abspath(image_path))
		else:
			hash_code = _md5_16(os.path.abspath(image_path))
	out_dir = os.path.join(output_root, hash_code)
	os.makedirs(out_dir, exist_ok=True)

	# 加载模型与图像（仅本地）
	pipe = _load_pipe(model_id, device, sd_local_path)
	watermarked_img = Image.open(image_path).convert('RGB')
	img_tensor = _preprocess_pil_to_tensor(watermarked_img, int(height), int(width), pipe.text_encoder.dtype, device)

	# VAE编码得到 Z0^w
	latents_0 = pipe.get_image_latents(img_tensor, sample=False)
	print(f"[Stage1] Z0^w 形状: {tuple(latents_0.shape)}")

	# DDIM反演（正向扩散到噪声方向）得到 Z_T^w
	text_embeddings, gs = _build_text_embeddings(pipe, prompt, guidance_scale, blind)
	latents_T = pipe.forward_diffusion(
		text_embeddings=text_embeddings,
		latents=latents_0,
		num_inference_steps=int(num_inference_steps),
		guidance_scale=float(gs),
	)
	print(f"[Stage1] Z_T^w 形状: {tuple(latents_T.shape)}")

	# 反解水印
	embedder = WatermarkEmbedder(
		wm_len=int(wm_len),
		center_interval_ratio=float(center_interval_ratio),
		shuffle_random_seed=int(shuffle_random_seed),
		encrypt_random_seed=int(encrypt_random_seed),
		tlt_intervals_num=int(tlt_intervals_num),
		device=device,
	)
	wm_repeat, reversed_tlt = embedder.deembedding_wm_tlt(latents_T)

	# 形状重塑到潜空间维度 (C,H,W)
	if latents_0.dim() == 4:
		_, c, h, w = latents_0.shape
	else:
		c, h, w = latents_0.shape
	latent_len = c * h * w
	assert wm_repeat.numel() == latent_len, f"wm_repeat长度不匹配: {wm_repeat.numel()} vs {latent_len}"
	assert reversed_tlt.size == latent_len, f"reversed_tlt长度不匹配: {reversed_tlt.size} vs {latent_len}"

	W_cop_tilde = wm_repeat.view(c, h, w).detach().to('cpu')
	W_loc_tilde = torch.from_numpy(reversed_tlt.astype(np.float32)).view(c, h, w)

	# 落盘
	z0_path = os.path.join(out_dir, f"z0_w_{hash_code}.pt")
	zT_path = os.path.join(out_dir, f"z_tw_{hash_code}.pt")
	wcop_path = os.path.join(out_dir, f"w_cop_tilde_{hash_code}.pt")
	wloc_path = os.path.join(out_dir, f"w_loc_tilde_{hash_code}.pt")
	torch.save(latents_0.detach().to('cpu'), z0_path)
	torch.save(latents_T.detach().to('cpu'), zT_path)
	torch.save(W_cop_tilde, wcop_path)
	torch.save(W_loc_tilde, wloc_path)

	# 记录参数
	meta = {
		"image_path": os.path.abspath(image_path),
		"device": device,
		"model_id": model_id,
		"height": int(height),
		"width": int(width),
		"num_inference_steps": int(num_inference_steps),
		"guidance_scale_used": float(gs),
		"prompt": prompt or "",
		"blind": bool(blind),
		"wm_len": int(wm_len),
		"tlt_intervals_num": int(tlt_intervals_num),
		"center_interval_ratio": float(center_interval_ratio),
		"shuffle_random_seed": int(shuffle_random_seed),
		"encrypt_random_seed": int(encrypt_random_seed),
		"outputs": {
			"z0_w": z0_path,
			"z_t_w": zT_path,
			"w_cop_tilde": wcop_path,
			"w_loc_tilde": wloc_path,
		},
	}
	with open(os.path.join(out_dir, f"stage1_args_{hash_code}.json"), 'w', encoding='utf-8') as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	print(f"[Stage1] 已保存 Z0^w: {z0_path}")
	print(f"[Stage1] 已保存 Z_T^w: {zT_path}")
	print(f"[Stage1] 已保存 ~W_cop: {wcop_path}")
	print(f"[Stage1] 已保存 ~W_loc: {wloc_path}")

	return {
		"hash": hash_code,
		"z0_w_path": z0_path,
		"z_t_w_path": zT_path,
		"w_cop_tilde_path": wcop_path,
		"w_loc_tilde_path": wloc_path,
		"out_dir": out_dir,
	}


def _resolve_default_image(hash_code: str) -> Optional[str]:
	base = os.path.join(os.path.dirname(__file__), '..', 'watermarkLOC', 'output', hash_code)
	base = os.path.abspath(base)
	cand = os.path.join(base, f"image_{hash_code}.png")
	return cand if os.path.exists(cand) else None


def main():
	parser = argparse.ArgumentParser(description='Stage1: 隐空间重构与水印初步解码')
	parser.add_argument('--image_path', type=str, default='')
	parser.add_argument('--hash', type=str, default='')
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
	parser.add_argument('--sd_local_path', type=str, default='')
	parser.add_argument('--height', type=int, default=512)
	parser.add_argument('--width', type=int, default=512)
	parser.add_argument('--num_inference_steps', type=int, default=50)
	parser.add_argument('--guidance_scale', type=float, default=1.0)
	parser.add_argument('--prompt', type=str, default='')
	parser.add_argument('--blind', action='store_true', default=True)
	parser.add_argument('--wm_len', type=int, default=256)
	parser.add_argument('--tlt_intervals_num', type=int, default=3)
	parser.add_argument('--center_interval_ratio', type=float, default=0.5)
	parser.add_argument('--shuffle_random_seed', type=int, default=133563)
	parser.add_argument('--encrypt_random_seed', type=int, default=133563)
	parser.add_argument('--output_root', type=str, default='')
	args = parser.parse_args()

	# 解析输入图像
	image_path = args.image_path.strip()
	hash_code = args.hash.strip()
	if not image_path:
		assert hash_code, "未提供 --image_path；请提供 --hash 以从 watermarkLOC/output/<hash>/ 自动定位图像"
		cand = _resolve_default_image(hash_code)
		assert cand is not None and os.path.exists(cand), f"未找到默认图像: {cand}"
		image_path = cand

	run_stage1(
		image_path=image_path,
		device=args.device,
		model_id=args.model_id,
		height=int(args.height),
		width=int(args.width),
		num_inference_steps=int(args.num_inference_steps),
		guidance_scale=float(args.guidance_scale),
		prompt=(args.prompt if args.prompt.strip() else None),
		blind=bool(args.blind),
		wm_len=int(args.wm_len),
		tlt_intervals_num=int(args.tlt_intervals_num),
		center_interval_ratio=float(args.center_interval_ratio),
		shuffle_random_seed=int(args.shuffle_random_seed),
		encrypt_random_seed=int(args.encrypt_random_seed),
		output_root=(args.output_root.strip() or None),
		hash_code=(hash_code or None),
		sd_local_path=(args.sd_local_path.strip() or None),
	)


if __name__ == '__main__':
	main() 