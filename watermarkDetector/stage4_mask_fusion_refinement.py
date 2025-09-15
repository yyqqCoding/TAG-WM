import os
import sys
import argparse
import json
from typing import Optional, Dict, Any

import torch

# 确保可从仓库根目录导入
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
	sys.path.append(REPO_ROOT)

# 可选导入DVRD
try:
	from DVRD import api as DVRD_API
except Exception:
	DVRD_API = None


def _ensure_out_dir(hash_code: str, output_root: Optional[str]) -> str:
	root = output_root or os.path.join(os.path.dirname(__file__), 'output')
	out_dir = os.path.join(root, hash_code)
	os.makedirs(out_dir, exist_ok=True)
	return out_dir


def _resolve_inputs(hash_code: str,
				   m_phys_latent_path: Optional[str],
				   m_sem_latent_path: Optional[str],
				   m_phys_img_path: Optional[str],
				   m_sem_img_path: Optional[str]) -> Dict[str, str]:
	base = os.path.join(os.path.dirname(__file__), 'output', hash_code)
	paths = {
		"M_phys_latent": m_phys_latent_path or os.path.join(base, f"M_phys_latent_{hash_code}.pt"),
		"M_sem_latent": m_sem_latent_path or os.path.join(base, f"M_sem_latent_{hash_code}.pt"),
		"M_phys_image": m_phys_img_path or os.path.join(base, f"M_phys_image_{hash_code}.pt"),
		"M_sem_image": m_sem_img_path or os.path.join(base, f"M_sem_image_{hash_code}.pt"),
	}
	for k, p in paths.items():
		assert os.path.exists(p), f"缺少输入掩码 {k}: {p}"
	return paths


def _binary_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	return ((a.int() + b.int()) > 0).int()


def _refine_latent(mask_latent_chw: torch.Tensor,
				  method: str,
				  dvrd_checkpoint: Optional[str],
				  train_size: int,
				  device: str) -> torch.Tensor:
	"""
	使用DVRD进行可选精炼：
	- trainable：加载UNet权重，输入/输出均为 [1,4,H,W]；
	- trainfree：对单通道聚合成 [1,1,H,W] 后进行多尺度阈值化，再复制到4通道。
	"""
	if method not in ('none', 'trainable', 'trainfree'):
		raise ValueError(f"未知精炼方法: {method}")
	if method == 'none' or DVRD_API is None:
		return mask_latent_chw

	C, H, W = mask_latent_chw.shape
	mask_batch = mask_latent_chw.unsqueeze(0).to(device)
	if method == 'trainable':
		assert dvrd_checkpoint and os.path.exists(dvrd_checkpoint), f"DVRD权重不存在: {dvrd_checkpoint}"
		model = DVRD_API.from_pretrained(
			checkpoint_path=dvrd_checkpoint,
			train_size=train_size,
			torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
			strict=False,
			device=device,
		)
		with torch.no_grad():
			refined = model(mask_batch.half() if device.startswith('cuda') else mask_batch.float())
		refined = (refined >= 0.5).int()[0]  # (4,H,W)
		return refined.cpu()
	else:
		# trainfree: 多尺度聚合阈值化
		aggregated = mask_batch.float().mean(dim=1, keepdim=True)  # [1,1,H,W]
		model = DVRD_API.TrainfreeDVRD(max_kernel_size=None, adaptive_max_kernel_size=True, overlapping=False)
		with torch.no_grad():
			refined_1ch = model(aggregated, confidence=0.5)  # [1,1,H,W] int
		refined_1ch = refined_1ch.to(torch.int)[0, 0]
		refined_4ch = refined_1ch.unsqueeze(0).repeat(4, 1, 1)
		return refined_4ch.cpu()


@torch.inference_mode()
def run_stage4(
	*,
	hash_code: str,
	device: str = 'cuda',
	m_phys_latent_path: Optional[str] = None,
	m_sem_latent_path: Optional[str] = None,
	m_phys_img_path: Optional[str] = None,
	m_sem_img_path: Optional[str] = None,
	refine_method: str = 'none',  # 'none' | 'trainfree' | 'trainable'
	dvrd_checkpoint: Optional[str] = None,
	dvrd_train_size: int = 512,
	output_root: Optional[str] = None,
) -> Dict[str, Any]:
	"""
	阶段四：掩码融合与精炼
	- 输入：阶段3生成的 M_phys 与 M_sem（潜/图像空间）
	- 融合：OR -> M_final_raw
	- 精炼：可选DVRD -> M_final
	- 输出：保存潜/图像空间最终掩码
	"""
	device = device if torch.cuda.is_available() and str(device).startswith('cuda') else 'cpu'
	out_dir = _ensure_out_dir(hash_code, output_root)
	paths = _resolve_inputs(hash_code, m_phys_latent_path, m_sem_latent_path, m_phys_img_path, m_sem_img_path)

	# 读取掩码
	M_phys_latent: torch.Tensor = torch.load(paths['M_phys_latent'], map_location='cpu')  # (4,H,W) int/float
	M_sem_latent: torch.Tensor = torch.load(paths['M_sem_latent'], map_location='cpu')
	M_phys_img: torch.Tensor = torch.load(paths['M_phys_image'], map_location='cpu')    # (H,W) int
	M_sem_img: torch.Tensor = torch.load(paths['M_sem_image'], map_location='cpu')

	# 统一为int二值
	M_phys_latent = (M_phys_latent > 0.5).int()
	M_sem_latent = (M_sem_latent > 0.5).int()
	M_phys_img = (M_phys_img > 0.5).int()
	M_sem_img = (M_sem_img > 0.5).int()

	# 融合
	M_final_latent_raw = _binary_or(M_phys_latent, M_sem_latent)
	M_final_img_raw = _binary_or(M_phys_img, M_sem_img)

	# 精炼（潜空间）
	M_final_latent = _refine_latent(M_final_latent_raw, refine_method, dvrd_checkpoint, dvrd_train_size, device)
	# 图像空间可直接保留 OR 结果（或后续扩展图像域精炼）
	M_final_img = M_final_img_raw

	# 保存
	m_final_latent_path = os.path.join(out_dir, f"M_final_latent_{hash_code}.pt")
	m_final_img_path = os.path.join(out_dir, f"M_final_image_{hash_code}.pt")
	torch.save(M_final_latent, m_final_latent_path)
	torch.save(M_final_img, m_final_img_path)

	# 记录
	meta = {
		"hash": hash_code,
		"refine_method": refine_method,
		"dvrd_checkpoint": (os.path.abspath(dvrd_checkpoint) if dvrd_checkpoint else None),
		"inputs": paths,
		"outputs": {
			"M_final_latent": m_final_latent_path,
			"M_final_image": m_final_img_path,
		},
	}
	with open(os.path.join(out_dir, f"stage4_args_{hash_code}.json"), 'w', encoding='utf-8') as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	print(f"[Stage4] M_final_latent 保存: {m_final_latent_path}")
	print(f"[Stage4] M_final_image 保存: {m_final_img_path}")

	return {"M_final_latent": m_final_latent_path, "M_final_image": m_final_img_path, "out_dir": out_dir}


def main():
	parser = argparse.ArgumentParser(description='Stage4: 掩码融合与精炼')
	parser.add_argument('--hash', type=str, required=True)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--m_phys_latent_path', type=str, default='')
	parser.add_argument('--m_sem_latent_path', type=str, default='')
	parser.add_argument('--m_phys_img_path', type=str, default='')
	parser.add_argument('--m_sem_img_path', type=str, default='')
	parser.add_argument('--refine_method', type=str, default='none', choices=['none', 'trainfree', 'trainable'])
	parser.add_argument('--dvrd_checkpoint', type=str, default='')
	parser.add_argument('--dvrd_train_size', type=int, default=512)
	parser.add_argument('--output_root', type=str, default='')
	args = parser.parse_args()

	run_stage4(
		hash_code=args.hash.strip(),
		device=args.device,
		m_phys_latent_path=(args.m_phys_latent_path.strip() or None),
		m_sem_latent_path=(args.m_sem_latent_path.strip() or None),
		m_phys_img_path=(args.m_phys_img_path.strip() or None),
		m_sem_img_path=(args.m_sem_img_path.strip() or None),
		refine_method=args.refine_method,
		dvrd_checkpoint=(args.dvrd_checkpoint.strip() or None),
		dvrd_train_size=int(args.dvrd_train_size),
		output_root=(args.output_root.strip() or None),
	)


if __name__ == '__main__':
	main() 