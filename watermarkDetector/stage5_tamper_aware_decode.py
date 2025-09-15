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

# 复用嵌入器
try:
	from applied_to_sd2.watermark_embedder import WatermarkEmbedder
except Exception:
	from watermark_embedder import WatermarkEmbedder


def _ensure_out_dir(hash_code: str, output_root: Optional[str]) -> str:
	root = output_root or os.path.join(os.path.dirname(__file__), 'output')
	out_dir = os.path.join(root, hash_code)
	os.makedirs(out_dir, exist_ok=True)
	return out_dir


def _resolve_inputs(hash_code: str,
				   wcop_tilde_path: Optional[str],
				   m_final_latent_path: Optional[str]) -> Dict[str, str]:
	base = os.path.join(os.path.dirname(__file__), 'output', hash_code)
	paths = {
		"W_cop_tilde": wcop_tilde_path or os.path.join(base, f"w_cop_tilde_{hash_code}.pt"),
		"M_final_latent": m_final_latent_path or os.path.join(base, f"M_final_latent_{hash_code}.pt"),
	}
	for k, p in paths.items():
		assert os.path.exists(p), f"缺少输入 {k}: {p}"
	return paths


@torch.inference_mode()
def run_stage5(
	*,
	hash_code: str,
	wm_len: int = 256,
	device: str = 'cuda',
	wcop_tilde_path: Optional[str] = None,
	m_final_latent_path: Optional[str] = None,
	center_interval_ratio: float = 0.5,
	shuffle_random_seed: int = 133563,
	encrypt_random_seed: int = 133563,
	tlt_intervals_num: int = 3,
	output_root: Optional[str] = None,
) -> Dict[str, Any]:
	"""
	阶段五：篡改感知版权解码
	- 输入：~W_cop（(C,H,W)）、M_final_latent（(C,H,W)）
	- 过程：将M_final作为不篡改置信度的反向权重指导投票，恢复版权消息
	- 输出：恢复版权消息比特并保存
	"""
	device = device if torch.cuda.is_available() and str(device).startswith('cuda') else 'cpu'
	out_dir = _ensure_out_dir(hash_code, output_root)
	paths = _resolve_inputs(hash_code, wcop_tilde_path, m_final_latent_path)

	# 加载输入
	W_cop_tilde: torch.Tensor = torch.load(paths['W_cop_tilde'], map_location='cpu')  # (C,H,W), float 0/1
	M_final_latent: torch.Tensor = torch.load(paths['M_final_latent'], map_location='cpu')  # (C,H,W), int 0/1
	if W_cop_tilde.dim() == 3:
		C, H, W = W_cop_tilde.shape
	else:
		raise ValueError('W_cop_tilde 需要为 (C,H,W)')

	# 展平
	wm_repeat = W_cop_tilde.view(-1).to(device).float()
	mask_pre_shuffle = M_final_latent.view(-1).to(device).float()  # 当前为"预打乱"域

	# 构造嵌入器（与嵌入/解码保持相同参数）
	embedder = WatermarkEmbedder(
		wm_len=int(wm_len),
		center_interval_ratio=float(center_interval_ratio),
		shuffle_random_seed=int(shuffle_random_seed),
		encrypt_random_seed=int(encrypt_random_seed),
		tlt_intervals_num=int(tlt_intervals_num),
		device=device,
	)

	# 将掩码打乱至潜变量域，以匹配 embedder.calc_watermark 内部的 inverse_shuffle 流程
	mask_shuffled = embedder.shuffle(mask_pre_shuffle)

	# 计算加权多数投票
	recovered = embedder.calc_watermark(
		wm_len=int(wm_len),
		wm_repeat=wm_repeat,
		pred_tamper_loc_latent=mask_shuffled,
		with_tamper_loc=True,
	)
	# 保存
	msg_bits_path = os.path.join(out_dir, f"copyright_bits_{hash_code}.pt")
	torch.save(recovered.cpu(), msg_bits_path)
	# 文本化（0/1字符串与十六进制）
	bits_str = ''.join([str(int(b.item())) for b in recovered])
	hex_str = hex(int(bits_str, 2))[2:].zfill((len(bits_str) + 3) // 4)
	msg_txt_path = os.path.join(out_dir, f"copyright_bits_{hash_code}.txt")
	with open(msg_txt_path, 'w', encoding='utf-8') as f:
		f.write(bits_str + '\n')
		f.write(hex_str + '\n')

	# 记录
	meta = {
		"hash": hash_code,
		"wm_len": int(wm_len),
		"center_interval_ratio": float(center_interval_ratio),
		"shuffle_random_seed": int(shuffle_random_seed),
		"encrypt_random_seed": int(encrypt_random_seed),
		"tlt_intervals_num": int(tlt_intervals_num),
		"inputs": paths,
		"outputs": {
			"bits_tensor": msg_bits_path,
			"bits_text": msg_txt_path,
			"bits_hex": hex_str,
		},
	}
	with open(os.path.join(out_dir, f"stage5_args_{hash_code}.json"), 'w', encoding='utf-8') as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	print(f"[Stage5] 版权消息比特保存: {msg_bits_path}")
	print(f"[Stage5] 文本输出: {msg_txt_path}")
	print(f"[Stage5] HEX: {hex_str}")

	return {"bits_tensor": msg_bits_path, "bits_text": msg_txt_path, "hex": hex_str, "out_dir": out_dir}


def main():
	parser = argparse.ArgumentParser(description='Stage5: 篡改感知版权解码（单一接口）')
	parser.add_argument('--hash', type=str, required=True)
	parser.add_argument('--wm_len', type=int, default=256)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--wcop_tilde_path', type=str, default='')
	parser.add_argument('--m_final_latent_path', type=str, default='')
	parser.add_argument('--center_interval_ratio', type=float, default=0.5)
	parser.add_argument('--shuffle_random_seed', type=int, default=133563)
	parser.add_argument('--encrypt_random_seed', type=int, default=133563)
	parser.add_argument('--tlt_intervals_num', type=int, default=3)
	parser.add_argument('--output_root', type=str, default='')
	args = parser.parse_args()

	run_stage5(
		hash_code=args.hash.strip(),
		wm_len=int(args.wm_len),
		device=args.device,
		wcop_tilde_path=(args.wcop_tilde_path.strip() or None),
		m_final_latent_path=(args.m_final_latent_path.strip() or None),
		center_interval_ratio=float(args.center_interval_ratio),
		shuffle_random_seed=int(args.shuffle_random_seed),
		encrypt_random_seed=int(args.encrypt_random_seed),
		tlt_intervals_num=int(args.tlt_intervals_num),
		output_root=(args.output_root.strip() or None),
	)


if __name__ == '__main__':
	main() 