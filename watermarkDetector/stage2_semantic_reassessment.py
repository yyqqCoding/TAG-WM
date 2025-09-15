import os
import sys
import argparse
import json
from typing import Optional, Dict, Any, List, Tuple

import torch
from PIL import Image

# 仅使用本地模型
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

# 确保可从仓库根目录导入
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
	sys.path.append(REPO_ROOT)
# 同时将 watermarkLOC 目录加入 sys.path，便于其内部模块使用顶级导入（model_config）
LOC_ROOT = os.path.join(REPO_ROOT, 'watermarkLOC')
if LOC_ROOT not in sys.path:
	sys.path.append(LOC_ROOT)

# 复用工具
try:
	from watermarkLOC.caption_utils import generate_caption
	from watermarkLOC.patch_utils import extract_patch_from_image, visualize_patch_grid
	except_msg = None
except Exception as e:
	except_msg = str(e)
	from caption_utils import generate_caption
	from patch_utils import extract_patch_from_image, visualize_patch_grid

# 本地路径解析
try:
	from watermarkLOC.model_config import get_model_path
except Exception:
	def get_model_path(model_name: str) -> str:
		return model_name

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer


def _resolve_path_with_env(candidates: List[str]) -> Optional[str]:
	for p in candidates:
		if p and os.path.isdir(p):
			return p
	return None


def _resolve_vlm_local_path(model_id: str, vlm_local_path: Optional[str]) -> str:
	cands: List[str] = []
	if vlm_local_path and vlm_local_path.strip():
		cands.append(vlm_local_path.strip())
	c = get_model_path(model_id)
	if c and c != model_id:
		cands.append(c)
	short = model_id.split('/')[-1] if '/' in model_id else model_id
	c2 = get_model_path(short)
	if c2 and c2 not in cands:
		cands.append(c2)
	# 环境变量
	for k in ['BLIP2_FLAN_T5_XL_PATH', 'BLIP2_LOCAL_PATH']:
		v = os.getenv(k, '').strip()
		if v:
			cands.append(v)
	p = _resolve_path_with_env(cands)
	if p is None:
		raise FileNotFoundError(f"未找到BLIP-2本地目录；候选: {cands}")
	return p


def _resolve_sent_local_path(model_id: str, sent_local_path: Optional[str]) -> str:
	cands: List[str] = []
	if sent_local_path and sent_local_path.strip():
		cands.append(sent_local_path.strip())
	c = get_model_path(model_id)
	if c and c != model_id:
		cands.append(c)
	short = model_id.split('/')[-1] if '/' in model_id else model_id
	c2 = get_model_path(short)
	if c2 and c2 not in cands:
		cands.append(c2)
	# 环境变量
	for k in ['SENTENCE_TRANSFORMER_PATH', 'SENT_LOCAL_PATH']:
		v = os.getenv(k, '').strip()
		if v:
			cands.append(v)
	p = _resolve_path_with_env(cands)
	if p is None:
		raise FileNotFoundError(f"未找到SentenceTransformer本地目录；候选: {cands}")
	return p


def _resolve_default_image(hash_code: str) -> Optional[str]:
	base = os.path.join(os.path.dirname(__file__), '..', 'watermarkLOC', 'output', hash_code)
	base = os.path.abspath(base)
	cand = os.path.join(base, f"image_{hash_code}.png")
	return cand if os.path.exists(cand) else None


def _ensure_out_dir(hash_code: str, output_root: Optional[str]) -> str:
	root = output_root or os.path.join(os.path.dirname(__file__), 'output')
	out_dir = os.path.join(root, hash_code)
	os.makedirs(out_dir, exist_ok=True)
	return out_dir


def _load_models_local(
	device: str,
	vlm_model_id: str,
	vlm_local_path: Optional[str],
	sentence_model_id: str,
	sent_local_path: Optional[str],
) -> Tuple[Blip2Processor, Blip2ForConditionalGeneration, SentenceTransformer]:
	vlm_path = _resolve_vlm_local_path(vlm_model_id, vlm_local_path)
	print(f"[Stage2] 本地加载VLM: {vlm_model_id} @ {vlm_path}")
	processor = Blip2Processor.from_pretrained(vlm_path, local_files_only=True)
	vlm = Blip2ForConditionalGeneration.from_pretrained(
		vlm_path,
		torch_dtype=(torch.float16 if device.startswith('cuda') else torch.float32),
		local_files_only=True,
	).to(device)

	sent_path = _resolve_sent_local_path(sentence_model_id, sent_local_path)
	print(f"[Stage2] 本地加载SentenceTransformer: {sentence_model_id} @ {sent_path}")
	sent_model = SentenceTransformer(sent_path).to(device)
	return processor, vlm, sent_model


@torch.inference_mode()
def run_stage2(
	*,
	hash_code: Optional[str] = None,
	image_path: Optional[str] = None,
	device: str = 'cuda',
	patch_grid_size: int = 8,
	vlm_model_id: str = 'Salesforce/blip2-flan-t5-xl',
	sentence_model_id: str = 'kasraarabi/finetuned-caption-embedding',
	vlm_local_path: Optional[str] = None,
	sent_local_path: Optional[str] = None,
	output_root: Optional[str] = None,
	save_grid_vis: bool = False,
	save_patch_crops: bool = False,
	log_captions: bool = False,
	max_patch_crops: int = 8,
) -> Dict[str, Any]:
	"""
	阶段二：当前语义重评估
	- 输入：图像（对应阶段1的水印图）
	- 处理：按 patch_grid_size 网格逐补丁生成caption并编码为语义向量
	- 输出：当前语义地图（num_patches x D）到 watermarkDetector/output/<hash>/semantic_map_current_<hash>.pt
	"""
	device = device if torch.cuda.is_available() and str(device).startswith('cuda') else 'cpu'

	# 解析图像路径
	if not image_path:
		assert hash_code, "未提供 --image_path；请提供 --hash 用于自动定位图像"
		cand = _resolve_default_image(hash_code)
		assert cand is not None and os.path.exists(cand), f"未找到默认图像: {cand}"
		image_path = cand
	else:
		assert os.path.exists(image_path), f"图像不存在: {image_path}"
	if not hash_code:
		# 从文件名中推断hash
		bn = os.path.basename(image_path)
		if bn.startswith('image_') and bn.endswith('.png'):
			hash_code = bn[len('image_'):-len('.png')]
		else:
			hash_code = os.path.splitext(bn)[0]

	out_dir = _ensure_out_dir(hash_code, output_root)

	# 加载模型（本地）
	processor, vlm, sent_model = _load_models_local(
		device=device,
		vlm_model_id=vlm_model_id,
		vlm_local_path=vlm_local_path,
		sentence_model_id=sentence_model_id,
		sent_local_path=sent_local_path,
	)

	# 打开图像
	image = Image.open(image_path).convert('RGB')

	# 可选保存网格可视化
	captions_log: List[str] = []
	if save_grid_vis:
		grid_img = visualize_patch_grid(image, patch_grid_size)
		grid_path = os.path.join(out_dir, f"grid_{hash_code}.png")
		grid_img.save(grid_path)
		print(f"[Stage2] 已保存网格可视化: {grid_path}")

	# 逐补丁生成语义向量
	num_patches = patch_grid_size * patch_grid_size
	semantic_vectors: List[torch.Tensor] = []
	for patch_idx in range(num_patches):
		patch_img = extract_patch_from_image(image, patch_idx, patch_grid_size)
		caption = generate_caption(patch_img, processor, vlm, device=device)
		emb = sent_model.encode(caption, convert_to_tensor=True).to(device)
		emb = emb / torch.norm(emb)
		semantic_vectors.append(emb)
		if log_captions:
			captions_log.append(f"patch {patch_idx:02d}: {caption}")
		if save_patch_crops and patch_idx < int(max_patch_crops):
			patch_dir = os.path.join(out_dir, 'patches')
			os.makedirs(patch_dir, exist_ok=True)
			patch_img.save(os.path.join(patch_dir, f"patch_{patch_idx:02d}.png"))

	print(f"[Stage2] 提取得到 {len(semantic_vectors)} 个语义向量")
	semantic_map = torch.stack(semantic_vectors).detach().cpu()
	sem_path = os.path.join(out_dir, f"semantic_map_current_{hash_code}.pt")
	torch.save(semantic_map, sem_path)
	print(f"[Stage2] 当前语义地图已保存: {sem_path}")

	cap_path = ''
	if log_captions and len(captions_log) > 0:
		cap_path = os.path.join(out_dir, f"captions_{hash_code}.txt")
		with open(cap_path, 'w', encoding='utf-8') as f:
			f.write('\n'.join(captions_log))
		print(f"[Stage2] Captions日志已保存: {cap_path}")

	# 记录元数据
	meta = {
		"hash": hash_code,
		"image_path": os.path.abspath(image_path),
		"device": device,
		"patch_grid_size": int(patch_grid_size),
		"vlm_model_id": vlm_model_id,
		"sentence_model_id": sentence_model_id,
		"outputs": {
			"semantic_map": sem_path,
			"captions": (cap_path or None),
		},
	}
	with open(os.path.join(out_dir, f"stage2_args_{hash_code}.json"), 'w', encoding='utf-8') as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	return {"semantic_map_path": sem_path, "captions_path": (cap_path or None), "out_dir": out_dir}


def main():
	parser = argparse.ArgumentParser(description='Stage2: 当前语义重评估')
	parser.add_argument('--hash', type=str, default='')
	parser.add_argument('--image_path', type=str, default='')
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--patch_grid_size', type=int, default=8)
	parser.add_argument('--vlm_model_id', type=str, default='Salesforce/blip2-flan-t5-xl')
	parser.add_argument('--vlm_local_path', type=str, default='')
	parser.add_argument('--sentence_model_id', type=str, default='kasraarabi/finetuned-caption-embedding')
	parser.add_argument('--sent_local_path', type=str, default='')
	parser.add_argument('--output_root', type=str, default='')
	parser.add_argument('--save_grid_vis', action='store_true', default=False)
	parser.add_argument('--save_patch_crops', action='store_true', default=False)
	parser.add_argument('--log_captions', action='store_true', default=False)
	parser.add_argument('--max_patch_crops', type=int, default=8)
	args = parser.parse_args()

	image_path = args.image_path.strip() or None
	hash_code = args.hash.strip() or None

	run_stage2(
		hash_code=hash_code,
		image_path=image_path,
		device=args.device,
		patch_grid_size=int(args.patch_grid_size),
		vlm_model_id=args.vlm_model_id,
		vlm_local_path=(args.vlm_local_path.strip() or None),
		sentence_model_id=args.sentence_model_id,
		sent_local_path=(args.sent_local_path.strip() or None),
		output_root=(args.output_root.strip() or None),
		save_grid_vis=bool(args.save_grid_vis),
		save_patch_crops=bool(args.save_patch_crops),
		log_captions=bool(args.log_captions),
		max_patch_crops=int(args.max_patch_crops),
	)


if __name__ == '__main__':
	main() 