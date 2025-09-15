import os
import sys
import argparse
import json
from typing import Optional, Dict, Any, Tuple

import torch
import numpy as np

# 确保可从仓库根目录导入
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
	sys.path.append(REPO_ROOT)
LOC_ROOT = os.path.join(REPO_ROOT, 'watermarkLOC')
if LOC_ROOT not in sys.path:
	sys.path.append(LOC_ROOT)

# 依赖工具（用于网格与可视化映射）
try:
	from watermarkLOC.patch_utils import map_latent_to_image_coords
except Exception:
	from patch_utils import map_latent_to_image_coords


def _ensure_out_dir(hash_code: str, output_root: Optional[str]) -> str:
	root = output_root or os.path.join(os.path.dirname(__file__), 'output')
	out_dir = os.path.join(root, hash_code)
	os.makedirs(out_dir, exist_ok=True)
	return out_dir


def _resolve_paths(hash_code: str,
				  w_loc_s_path: Optional[str],
				  w_loc_tilde_path: Optional[str],
				  orig_sem_path: Optional[str],
				  curr_sem_path: Optional[str]) -> Tuple[str, str, str, str]:
	"""
	从默认目录结构推断路径，或使用显式传入的路径。
	- 生成端（watermarkLOC/output/<hash>/）: w_loc_s_<hash>.pt, semantic_map_<hash>.pt
	- 检测端（watermarkDetector/output/<hash>/）: w_loc_tilde_<hash>.pt, semantic_map_current_<hash>.pt
	"""
	# 默认根
	gen_root = os.path.join(LOC_ROOT, 'output', hash_code)
	det_root = os.path.join(os.path.dirname(__file__), 'output', hash_code)

	if not w_loc_s_path:
		w_loc_s_path = os.path.join(gen_root, f'w_loc_s_{hash_code}.pt')
	if not w_loc_tilde_path:
		w_loc_tilde_path = os.path.join(det_root, f'w_loc_tilde_{hash_code}.pt')
	if not orig_sem_path:
		orig_sem_path = os.path.join(gen_root, f'semantic_map_{hash_code}.pt')
	if not curr_sem_path:
		curr_sem_path = os.path.join(det_root, f'semantic_map_current_{hash_code}.pt')

	assert os.path.exists(w_loc_s_path), f"缺少期望定位水印 W_loc^S: {w_loc_s_path}"
	assert os.path.exists(w_loc_tilde_path), f"缺少重构定位水印 ~W_loc: {w_loc_tilde_path}"
	assert os.path.exists(orig_sem_path), f"缺少原始语义地图: {orig_sem_path}"
	assert os.path.exists(curr_sem_path), f"缺少当前语义地图: {curr_sem_path}"
	return w_loc_s_path, w_loc_tilde_path, orig_sem_path, curr_sem_path


def _to_binary(t: torch.Tensor, thre: float = 0.5) -> torch.Tensor:
	return (t.float() >= thre).to(torch.int)


def _latent_mask_to_img(mask_latent_chw: torch.Tensor, image_size: int = 512) -> torch.Tensor:
	"""
	将潜空间掩码 (C,H,W) 映射为图像空间二值图 (H_img, W_img):
	- 先对通道取均值 -> (H,W)
	- 再最近邻上采样到 image_size。
	"""
	if mask_latent_chw.dim() == 3:
		c, h, w = mask_latent_chw.shape
	else:
		raise ValueError('mask_latent_chw 需要为 (C,H,W)')
	mask_hw = mask_latent_chw.float().mean(dim=0, keepdim=True)  # (1,H,W)
	mask_hw = (mask_hw >= 0.5).to(torch.float32)
	mask_img = torch.nn.functional.interpolate(mask_hw.unsqueeze(0), size=(image_size, image_size), mode='nearest').squeeze(0).squeeze(0)
	return (mask_img >= 0.5).to(torch.int)


def _build_patch_mask_img(patch_grid_size: int, patch_flags_hw: torch.Tensor, image_size: int = 512) -> torch.Tensor:
	"""
	根据 patch_grid_size 与每个patch的二值标记构建图像空间掩码（H_img,W_img）。
	patch_flags_hw: (H_grid, W_grid) 二值。
	"""
	H_grid, W_grid = patch_flags_hw.shape
	assert H_grid == patch_grid_size and W_grid == patch_grid_size

	cell = image_size // patch_grid_size
	mask = torch.zeros((image_size, image_size), dtype=torch.int)
	for i in range(patch_grid_size):
		for j in range(patch_grid_size):
			if int(patch_flags_hw[i, j]) == 1:
				y0, y1 = i * cell, (i + 1) * cell
				x0, x1 = j * cell, (j + 1) * cell
				mask[y0:y1, x0:x1] = 1
	return mask


def _build_patch_mask_latent(patch_grid_size: int, patch_flags_hw: torch.Tensor, latent_hw: Tuple[int, int]) -> torch.Tensor:
	"""
	根据 patch_grid_size 与每个patch的二值标记构建潜空间掩码（H_lat,W_lat）。
	返回 (H_lat, W_lat) int 掩码。
	"""
	H_lat, W_lat = latent_hw
	cell_h = H_lat // patch_grid_size
	cell_w = W_lat // patch_grid_size
	mask = torch.zeros((H_lat, W_lat), dtype=torch.int)
	for i in range(patch_grid_size):
		for j in range(patch_grid_size):
			if int(patch_flags_hw[i, j]) == 1:
				y0, y1 = i * cell_h, (i + 1) * cell_h
				x0, x1 = j * cell_w, (j + 1) * cell_w
				mask[y0:y1, x0:x1] = 1
	return mask


@torch.inference_mode()
def run_stage3(
	*,
	hash_code: str,
	patch_grid_size: int = 8,
	image_size: int = 512,
	sem_threshold: float = 0.25,
	w_loc_s_path: Optional[str] = None,
	w_loc_tilde_path: Optional[str] = None,
	orig_sem_path: Optional[str] = None,
	curr_sem_path: Optional[str] = None,
	output_root: Optional[str] = None,
) -> Dict[str, Any]:
	"""
	阶段三：SDD分析
	- 物理通道：XOR(W_loc^S, ~W_loc) -> M_phys
	- 语义通道：cos距离(v_i, \tilde{v}_i) > tau_sem -> M_sem
	输出潜空间(C,H,W)与图像空间(H_img,W_img)掩码。
	"""
	out_dir = _ensure_out_dir(hash_code, output_root)
	# 路径解析
	w_loc_s_path, w_loc_tilde_path, orig_sem_path, curr_sem_path = _resolve_paths(
		hash_code, w_loc_s_path, w_loc_tilde_path, orig_sem_path, curr_sem_path
	)

	# 加载张量
	W_loc_S: torch.Tensor = torch.load(w_loc_s_path, map_location='cpu')  # (C,H,W) 0/1
	W_loc_tilde: torch.Tensor = torch.load(w_loc_tilde_path, map_location='cpu')  # (C,H,W) 0/1
	assert W_loc_S.shape == W_loc_tilde.shape, f"形状不一致: {W_loc_S.shape} vs {W_loc_tilde.shape}"
	C, H_lat, W_lat = W_loc_S.shape

	orig_sem: torch.Tensor = torch.load(orig_sem_path, map_location='cpu')  # (N,D)
	curr_sem: torch.Tensor = torch.load(curr_sem_path, map_location='cpu')  # (N,D)
	assert orig_sem.shape == curr_sem.shape, f"语义维度不一致: {orig_sem.shape} vs {curr_sem.shape}"
	N, D = orig_sem.shape
	assert patch_grid_size * patch_grid_size == N, f"patch_grid_size^2 != 语义patch数: {patch_grid_size}^2 vs {N}"

	# 通道一：物理偏差（潜空间 XOR）
	A = _to_binary(W_loc_S)
	B = _to_binary(W_loc_tilde)
	M_phys_latent_chw = (A ^ B).to(torch.int)  # (C,H,W)
	M_phys_img_hw = _latent_mask_to_img(M_phys_latent_chw, image_size=image_size)  # (H_img,W_img)

	# 通道二：语义偏差（patch级阈值 -> 映射潜/图像）
	# 归一化，计算余弦距离
	orig_norm = orig_sem / (torch.norm(orig_sem, dim=1, keepdim=True) + 1e-8)
	curr_norm = curr_sem / (torch.norm(curr_sem, dim=1, keepdim=True) + 1e-8)
	cos_sim = (orig_norm * curr_norm).sum(dim=1)
	cos_dist = 1.0 - cos_sim  # (N,)
	flags = (cos_dist > float(sem_threshold)).to(torch.int)  # (N,)
	M_sem_grid = flags.view(patch_grid_size, patch_grid_size)  # (H_grid,W_grid)
	M_sem_img_hw = _build_patch_mask_img(patch_grid_size, M_sem_grid, image_size=image_size)
	M_sem_latent_hw = _build_patch_mask_latent(patch_grid_size, M_sem_grid, (H_lat, W_lat))
	M_sem_latent_chw = M_sem_latent_hw.unsqueeze(0).repeat(C, 1, 1)

	# 保存输出
	m_phys_latent_path = os.path.join(out_dir, f"M_phys_latent_{hash_code}.pt")
	m_phys_img_path = os.path.join(out_dir, f"M_phys_image_{hash_code}.pt")
	m_sem_latent_path = os.path.join(out_dir, f"M_sem_latent_{hash_code}.pt")
	m_sem_img_path = os.path.join(out_dir, f"M_sem_image_{hash_code}.pt")
	torch.save(M_phys_latent_chw, m_phys_latent_path)
	torch.save(M_phys_img_hw, m_phys_img_path)
	torch.save(M_sem_latent_chw, m_sem_latent_path)
	torch.save(M_sem_img_hw, m_sem_img_path)

	# 记录参数
	meta = {
		"hash": hash_code,
		"patch_grid_size": int(patch_grid_size),
		"image_size": int(image_size),
		"sem_threshold": float(sem_threshold),
		"inputs": {
			"W_loc_S": os.path.abspath(w_loc_s_path),
			"W_loc_tilde": os.path.abspath(w_loc_tilde_path),
			"orig_sem": os.path.abspath(orig_sem_path),
			"curr_sem": os.path.abspath(curr_sem_path),
		},
		"outputs": {
			"M_phys_latent": m_phys_latent_path,
			"M_phys_image": m_phys_img_path,
			"M_sem_latent": m_sem_latent_path,
			"M_sem_image": m_sem_img_path,
		},
	}
	with open(os.path.join(out_dir, f"stage3_args_{hash_code}.json"), 'w', encoding='utf-8') as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	print(f"[Stage3] M_phys_latent 保存: {m_phys_latent_path}")
	print(f"[Stage3] M_phys_image 保存: {m_phys_img_path}")
	print(f"[Stage3] M_sem_latent 保存: {m_sem_latent_path}")
	print(f"[Stage3] M_sem_image 保存: {m_sem_img_path}")

	return {
		"M_phys_latent": m_phys_latent_path,
		"M_phys_image": m_phys_img_path,
		"M_sem_latent": m_sem_latent_path,
		"M_sem_image": m_sem_img_path,
		"out_dir": out_dir,
	}


def main():
	parser = argparse.ArgumentParser(description='Stage3: 语义差分检测器（SDD）分析')
	parser.add_argument('--hash', type=str, required=True)
	parser.add_argument('--patch_grid_size', type=int, default=8)
	parser.add_argument('--image_size', type=int, default=512)
	parser.add_argument('--sem_threshold', type=float, default=0.25)
	parser.add_argument('--w_loc_s_path', type=str, default='')
	parser.add_argument('--w_loc_tilde_path', type=str, default='')
	parser.add_argument('--orig_sem_path', type=str, default='')
	parser.add_argument('--curr_sem_path', type=str, default='')
	parser.add_argument('--output_root', type=str, default='')
	args = parser.parse_args()

	run_stage3(
		hash_code=args.hash.strip(),
		patch_grid_size=int(args.patch_grid_size),
		image_size=int(args.image_size),
		sem_threshold=float(args.sem_threshold),
		w_loc_s_path=(args.w_loc_s_path.strip() or None),
		w_loc_tilde_path=(args.w_loc_tilde_path.strip() or None),
		orig_sem_path=(args.orig_sem_path.strip() or None),
		curr_sem_path=(args.curr_sem_path.strip() or None),
		output_root=(args.output_root.strip() or None),
	)


if __name__ == '__main__':
	main() 