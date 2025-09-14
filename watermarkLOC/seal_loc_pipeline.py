import os
import sys
import argparse
import hashlib
from typing import Optional, Tuple, Dict, Any, List

import torch
from PIL import Image
import numpy as np

# 环境与路径
os.environ.setdefault('DISABLE_DVRD', '1')

# 加入仓库根目录，使用包路径导入 applied_to_sd2/*
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
	sys.path.append(repo_root)

# 复用TAG-WM的可反演稳定扩散管线
try:
	from applied_to_sd2.inverse_stable_diffusion import InversableStableDiffusionPipeline
	from applied_to_sd2.modified_stable_diffusion import ModifiedStableDiffusionPipeline  # noqa: F401
except Exception as e:
	# 兜底：允许从diffusers直接加载标准管线（不影响阶段一需求）
	from diffusers import StableDiffusionPipeline as InversableStableDiffusionPipeline  # type: ignore
	print(f"[WARN] 使用标准StableDiffusionPipeline作为兜底: {e}")

# 复用TAG-WM嵌入器
try:
	from applied_to_sd2.watermark_embedder import WatermarkEmbedder
except Exception:
	from watermark_embedder import WatermarkEmbedder

# 模型配置
try:
	from model_config import get_model_path
except Exception:
	def get_model_path(model_name: str) -> str:
		return model_name

# 语义提取所需组件（本地/包路径双分支导入）
try:
	from caption_utils import generate_caption
	from patch_utils import extract_patch_from_image, visualize_patch_grid, map_latent_to_image_coords
	from simhash_utils import simhash_single_patch
except Exception:
	from watermarkLOC.caption_utils import generate_caption
	from watermarkLOC.patch_utils import extract_patch_from_image, visualize_patch_grid, map_latent_to_image_coords
	from watermarkLOC.simhash_utils import simhash_single_patch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from diffusers.schedulers import DDIMScheduler


class SEALLOCPipeline:
	"""
	SEAL-LOC 单文件流水线
	- 阶段一：代理生成（已完成）
	- 阶段二：VAE编码得到 Z0_pre（4x64x64）
	- 阶段三：逐补丁语义特征提取
	- 对外仅提供一个接口：generate_loc_watermark(prompt, until_stage)
	"""

	def __init__(
		self,
		device: str = 'cuda',
		model_id: str = 'stabilityai/stable-diffusion-2-1-base',
		height: int = 512,
		width: int = 512,
		patch_grid_size: int = 8,
		vlm_model_name: str = 'Salesforce/blip2-flan-t5-xl',
		sentence_model_name: str = 'kasraarabi/finetuned-caption-embedding',
		debug_stage3: bool = False,
		save_patch_grid_vis: bool = False,
		save_patch_crops: bool = False,
		log_patch_captions: bool = False,
		max_patch_crops: int = 8,
		simhash_bits: int = 7,
		wm_len: int = 256,
		tlt_intervals_num: int = 3,
		num_inference_steps: int = 50,
		guidance_scale: float = 7.5,
	):
		self.device = device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu'
		self.model_id = model_id
		height = int(height)
		width = int(width)
		self.height = height
		self.width = width
		self.pipe = None

		# 阶段二相关
		self.patch_grid_size = patch_grid_size
		self.vlm_model_name = vlm_model_name
		self.sentence_model_name = sentence_model_name
		self.vlm_processor: Optional[Blip2Processor] = None
		self.vlm_model: Optional[Blip2ForConditionalGeneration] = None
		self.sentence_model: Optional[SentenceTransformer] = None
		self.semantic_maps_dir = os.path.join(os.path.dirname(__file__), 'semantic_maps')
		os.makedirs(self.semantic_maps_dir, exist_ok=True)

		# 阶段三调试参数
		self.debug_stage3 = debug_stage3
		self.save_patch_grid_vis = save_patch_grid_vis
		self.save_patch_crops = save_patch_crops
		self.log_patch_captions = log_patch_captions
		self.max_patch_crops = int(max_patch_crops)

		# 阶段四相关
		self.simhash_bits = int(simhash_bits)
		self.stage4_dir = os.path.join(os.path.dirname(__file__), 'output', 'stage4')
		os.makedirs(self.stage4_dir, exist_ok=True)

		# 阶段五/六相关
		self.wm_len = int(wm_len)
		self.tlt_intervals_num = int(tlt_intervals_num)
		self.stage5_dir = os.path.join(os.path.dirname(__file__), 'output', 'stage5')
		self.stage6_dir = os.path.join(os.path.dirname(__file__), 'output', 'stage6')
		os.makedirs(self.stage5_dir, exist_ok=True)
		os.makedirs(self.stage6_dir, exist_ok=True)

		# 初始化嵌入器（复用TAG-WM）
		self.embedder = WatermarkEmbedder(
			wm_len=self.wm_len,
			center_interval_ratio=0.5,
			shuffle_random_seed=133563,
			encrypt_random_seed=133563,
			tlt_intervals_num=self.tlt_intervals_num,
			device=self.device,
		)
		# 阶段六生成图像参数
		self.num_inference_steps = int(num_inference_steps)
		self.guidance_scale = float(guidance_scale)

	def _load_diffusion_model(self):
		"""加载Stable Diffusion 2.1 Base，可反演管线（若可用）。"""
		if self.pipe is not None:
			return
		model_path = get_model_path(self.model_id)
		print(f"[Stage1] 加载扩散模型: {model_path}")
		try:
			self.pipe = InversableStableDiffusionPipeline.from_pretrained(
				model_path,
				torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
			)
		except Exception as e:
			print(f"[Stage1] 本地/指定路径加载失败（将尝试在线加载）: {e}")
			self.pipe = InversableStableDiffusionPipeline.from_pretrained(
				self.model_id,
				torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
			)
		# 禁用安全检查器（若存在）
		if hasattr(self.pipe, 'safety_checker'):
			self.pipe.safety_checker = None
		self.pipe = self.pipe.to(self.device)
		# 统一生成调度器为DDIM，确保与反演一致
		try:
			self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
		except Exception as _e:
			print(f"[Stage1] 切换DDIM调度器失败: {_e}")

	def _load_vlm_and_sentence_models(self):
		"""加载BLIP-2处理器与模型、SentenceTransformer（优先本地路径）。"""
		if self.vlm_processor is None or self.vlm_model is None:
			vlm_path = get_model_path(self.vlm_model_name)
			print(f"[Stage2] 加载VLM: {self.vlm_model_name} @ {vlm_path}")
			self.vlm_processor = Blip2Processor.from_pretrained(vlm_path, local_files_only=True)
			self.vlm_model = Blip2ForConditionalGeneration.from_pretrained(
				vlm_path,
				torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
				local_files_only=True,
			).to(self.device)
		if self.sentence_model is None:
			sent_path = get_model_path(self.sentence_model_name)
			print(f"[Stage2] 加载SentenceTransformer: {self.sentence_model_name} @ {sent_path}")
			self.sentence_model = SentenceTransformer(sent_path).to(self.device)

	@torch.inference_mode()
	def _stage1_proxy(self, prompt: str) -> Dict[str, Any]:
		"""
		阶段一：仅生成无水印代理图像 X_pre
		"""
		self._load_diffusion_model()
		print("[Stage1] 生成代理图像 X_pre ...")
		result = self.pipe(prompt=prompt, height=self.height, width=self.width)
		if hasattr(result, 'images'):
			image: Image.Image = result.images[0]
		else:
			image = result[0] if isinstance(result, (list, tuple)) else result
		return {
			"proxy_image": image,
		}

	def _infer_latent_grid_size(self, z0_pre: torch.Tensor) -> Tuple[int, int]:
		"""根据Z0_pre推断潜空间网格尺寸，如(64,64)或(28,28)。"""
		if z0_pre.dim() == 4:
			_, c, h, w = z0_pre.shape
		else:
			c, h, w = z0_pre.shape
		return h, w

	@torch.inference_mode()
	def _stage2_encode_latent(self, proxy_image: Image.Image) -> Dict[str, Any]:
		"""
		阶段二：对代理图像进行VAE编码，得到 Z0_pre (1,4,64,64)（512x512时）
		"""
		print("[Stage2] 编码代理图像到潜空间 Z0_pre ...")
		# 将PIL图像调整到生成尺寸并归一化到[-1,1]，再输入VAE编码器
		resample = getattr(Image, 'Resampling', Image)
		image_resized = proxy_image.resize((self.width, self.height), resample=getattr(resample, 'LANCZOS', Image.BICUBIC))
		image_np = np.array(image_resized).astype(np.float32) / 255.0
		image_np = image_np[None].transpose(0, 3, 1, 2)  # (1,3,H,W)
		image_tensor = torch.from_numpy(image_np).to(self.device, dtype=self.pipe.text_encoder.dtype)
		image_tensor = 2.0 * image_tensor - 1.0
		with torch.no_grad():
			encoding = self.pipe.vae.encode(image_tensor).latent_dist.sample()
		latents = encoding * 0.18215
		return {
			"Z0_pre": latents,
		}

	@torch.inference_mode()
	def _stage3_patch_semantics(self, proxy_image: Image.Image, z0_pre: torch.Tensor) -> Dict[str, Any]:
		"""
		阶段三：逐补丁语义特征提取
		- 将潜空间视为 patch_grid_size×patch_grid_size 网格
		- 通过图像空间裁剪对应区域，使用BLIP-2生成caption，再由SentenceTransformer编码
		- 返回并保存语义地图
		"""
		self._load_vlm_and_sentence_models()

		H, W = self._infer_latent_grid_size(z0_pre)
		print(f"[Stage3] 潜空间尺寸: {H}x{W}，网格: {self.patch_grid_size}x{self.patch_grid_size}")
		num_patches = self.patch_grid_size * self.patch_grid_size

		# 调试输出目录
		stage3_dir = os.path.join(os.path.dirname(__file__), 'output', 'stage3', self._current_prompt_hash)
		if self.debug_stage3:
			os.makedirs(stage3_dir, exist_ok=True)
			if self.save_patch_grid_vis:
				grid_img = visualize_patch_grid(proxy_image, self.patch_grid_size)
				grid_path = os.path.join(stage3_dir, f"grid_{self._current_prompt_hash}.png")
				grid_img.save(grid_path)
				print(f"[Stage3] 网格可视化已保存: {grid_path}")
			if self.save_patch_crops:
				patch_dir = os.path.join(stage3_dir, 'patches')
				os.makedirs(patch_dir, exist_ok=True)
			captions_log: List[str] = []

		semantic_vectors: List[torch.Tensor] = []
		for patch_idx in range(num_patches):
			patch_img = extract_patch_from_image(proxy_image, patch_idx, self.patch_grid_size)
			caption = generate_caption(patch_img, self.vlm_processor, self.vlm_model, device=self.device)
			embedding = self.sentence_model.encode(caption, convert_to_tensor=True).to(self.device)
			emb_norm = embedding / torch.norm(embedding)
			semantic_vectors.append(emb_norm)

			if self.debug_stage3:
				# 记录caption和坐标
				x0, y0, x1, y1 = map_latent_to_image_coords(patch_idx, self.patch_grid_size, (H, W), proxy_image.size[0])
				captions_log.append(f"patch {patch_idx:02d} @ ({x0},{y0},{x1},{y1}): {caption}")
				# 保存部分patch裁剪
				if self.save_patch_crops and patch_idx < self.max_patch_crops:
					patch_path = os.path.join(stage3_dir, 'patches', f"patch_{patch_idx:02d}.png")
					patch_img.save(patch_path)

		print(f"[Stage3] 提取得到 {len(semantic_vectors)} 个语义向量")
		semantic_map = torch.stack(semantic_vectors)  # (num_patches, D)
		prompt_hash = self._current_prompt_hash
		save_dir = self.semantic_maps_dir
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f"semantic_map_{prompt_hash}.pt")
		torch.save(semantic_map, save_path)
		print(f"[Stage3] 语义地图已保存: {save_path}")

		# 写入caption日志
		if self.debug_stage3 and self.log_patch_captions and len(captions_log) > 0:
			cap_log_path = os.path.join(stage3_dir, f"captions_{prompt_hash}.txt")
			with open(cap_log_path, 'w', encoding='utf-8') as f:
				f.write("\n".join(captions_log))
			print(f"[Stage3] Caption日志已保存: {cap_log_path}")

		return {
			"semantic_vectors": semantic_vectors,
			"semantic_map_path": save_path,
		}

	@torch.inference_mode()
	def _stage4_generate_wlocs(self, semantic_vectors: List[torch.Tensor], z0_pre: torch.Tensor) -> Dict[str, Any]:
		"""
		阶段四：基于逐patch语义→SimHash→确定性CSPRNG 生成动态语义定位水印 W_loc^S
		- 对每个patch计算SimHash，作为种子
		- 用基于SHA-256的确定性CSPRNG生成对应长度的二进制序列
		- 拼接得到与潜在空间同维度的 W_loc^S（0/1）
		"""
		# 潜空间尺寸
		if z0_pre.dim() == 4:
			_, c, h, w = z0_pre.shape
		else:
			c, h, w = z0_pre.shape
		latent_len = c * h * w
		num_patches = self.patch_grid_size * self.patch_grid_size
		bits_per_patch = latent_len // num_patches
		remaining = latent_len % num_patches

		def csprng_bits(seed_int: int, length_bits: int) -> np.ndarray:
			import hashlib as _hashlib
			bits = []
			counter = 0
			while len(bits) < length_bits:
				msg = f"{seed_int}:{counter}".encode()
				digest = _hashlib.sha256(msg).digest()
				for byte in digest:
					for i in range(8):
						bits.append((byte >> (7 - i)) & 1)
						if len(bits) >= length_bits:
							break
					if len(bits) >= length_bits:
						break
				counter += 1
			return np.array(bits, dtype=np.uint8)

		w_bits = np.zeros(latent_len, dtype=np.uint8)
		cursor = 0
		for i, emb in enumerate(semantic_vectors):
			seed = simhash_single_patch(emb, num_bits=self.simhash_bits, seed=42 + i)
			seg_len = bits_per_patch + (1 if (i == num_patches - 1 and remaining > 0) else 0)
			segment = csprng_bits(seed, seg_len)
			w_bits[cursor:cursor+seg_len] = segment
			cursor += seg_len

		# 形状为 (C,H,W) 的float张量（0/1）
		w_tensor = torch.from_numpy(w_bits.astype(np.float32)).to(self.device)
		w_tensor = w_tensor.view(c, h, w)

		# 落盘
		prompt_hash = self._current_prompt_hash
		out_path = os.path.join(self.stage4_dir, f"w_loc_s_{prompt_hash}.pt")
		torch.save(w_tensor, out_path)
		ones_ratio = float((w_tensor == 1).sum().item()) / float(w_tensor.numel())
		print(f"[Stage4] W_loc^S 生成完成: shape={tuple(w_tensor.shape)}, ones_ratio={ones_ratio:.6f}")
		print(f"[Stage4] 已保存: {out_path}")

		return {
			"W_loc_S": w_tensor,
			"W_loc_S_path": out_path,
		}

	@torch.inference_mode()
	def _stage5_generate_wcop(self) -> Dict[str, Any]:
		"""
		阶段五：版权水印生成 W_cop
		- 生成长度为 wm_len 的比特串（示例从prompt哈希确定性生成），并返回float张量（0/1）
		- 注意：生产中应从外部传入版权消息与密钥，这里按文档先就地实现
		"""
		ph = self._current_prompt_hash
		# 基于prompt哈希确定性生成wm_len位
		import hashlib as _hashlib
		bits = []
		counter = 0
		while len(bits) < self.wm_len:
			digest = _hashlib.sha256(f"{ph}:{counter}".encode()).digest()
			for byte in digest:
				for i in range(8):
					bits.append((byte >> (7 - i)) & 1)
					if len(bits) >= self.wm_len:
						break
				if len(bits) >= self.wm_len:
					break
			counter += 1
		wm_tensor = torch.tensor(bits, dtype=torch.float32, device=self.device)
		out_path = os.path.join(self.stage5_dir, f"w_cop_{ph}.pt")
		torch.save(w_tensor := wm_tensor, out_path)
		print(f"[Stage5] W_cop 生成完成: len={w_tensor.numel()}，已保存: {out_path}")
		return {"W_cop": w_tensor, "W_cop_path": out_path}

	@torch.inference_mode()
	def _stage6_dmjs_and_prepare(self, W_cop: torch.Tensor, W_loc_S: torch.Tensor, z0_pre: torch.Tensor) -> Dict[str, Any]:
		"""
		阶段六：双水印联合采样（DMJS）生成初始噪声 Z_T^w
		- 将 W_cop 展开/加密，与 W_loc^S (展平为tlt) 一起输入 embedder.embedding_wm_tlt
		- 输出 latent_noise: (1,C,H,W)
		- 本阶段仅生成噪声，不做最终扩散解码（留到后续扩展）
		"""
		# 形状
		if z0_pre.dim() == 4:
			_, c, h, w = z0_pre.shape
		else:
			c, h, w = z0_pre.shape
		latent_size = (c, h, w)
		latent_len = c * h * w

		# 准备 tlt：用 W_loc_S 展平作为定位比特
		tlt_bits = W_loc_S.view(-1).detach().to(torch.uint8).cpu().numpy()
		# 版权水印直接传入，由 embedder 完成展开、加密与采样
		latent_noise, wm_repeat = self.embedder.embedding_wm_tlt(W_cop, tlt_bits, latent_size)
		out_path = os.path.join(self.stage6_dir, f"z_tw_{self._current_prompt_hash}.pt")
		torch.save(latent_noise, out_path)
		print(f"[Stage6] Z_T^w 生成完成: shape={tuple(latent_noise.shape)}，已保存: {out_path}")
		return {"Z_T_w": latent_noise, "Z_T_w_path": out_path, "wm_repeat": wm_repeat}

	@torch.inference_mode()
	def _stage6b_generate_image(self, prompt: str, Z_T_w: torch.Tensor, num_inference_steps: int = 50, guidance_scale: float = 7.5) -> Dict[str, Any]:
		"""
		阶段六扩展：以 Z_T^w 作为初始latents进行标准去噪生成最终图像。
		"""
		self._load_diffusion_model()
		latents = Z_T_w.to(self.device).half() if self.device.startswith('cuda') else Z_T_w.to(self.device).float()
		result = self.pipe(
			prompt=prompt,
			height=self.height,
			width=self.width,
			num_inference_steps=num_inference_steps,
			guidance_scale=guidance_scale,
			latents=latents,
			output_type="pil",
			return_dict=True,
		)
		image: Image.Image = result.images[0]
		out_path = os.path.join(self.stage6_dir, f"image_{self._current_prompt_hash}.png")
		image.save(out_path)
		print(f"[Stage6] 最终图像已生成并保存: {out_path}")
		return {"final_image": image, "final_image_path": out_path}

	def generate_loc_watermark(self, prompt: str, until_stage: int = 1) -> Dict[str, Any]:
		"""
		对外统一接口：按照文档阶段顺序执行到 until_stage。
		"""
		if until_stage < 1 or until_stage > 6:
			raise ValueError("until_stage 必须在 [1,6] 范围内")

		outputs: Dict[str, Any] = {}
		self._current_prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16]
		# 统一输出目录
		base_out_dir = os.path.join(os.path.dirname(__file__), 'output', self._current_prompt_hash)
		os.makedirs(base_out_dir, exist_ok=True)

		# 阶段一：代理图像
		stage1 = self._stage1_proxy(prompt)
		outputs.update(stage1)
		print("[Stage1] 完成：已生成代理图像 X_pre")
		# 保存代理图
		proxy_path = os.path.join(base_out_dir, f"proxy_{self._current_prompt_hash}.png")
		outputs['proxy_image'].save(proxy_path)

		if until_stage == 1:
			return outputs

		# 阶段二：VAE编码
		stage2 = self._stage2_encode_latent(outputs['proxy_image'])
		outputs.update(stage2)
		print("[Stage2] 完成：已得到初始潜变量 Z0_pre")
		# 保存Z0信息
		z = outputs['Z0_pre']
		with open(os.path.join(base_out_dir, 'Z0_pre.txt'), 'w', encoding='utf-8') as f:
			f.write(f"Z0_pre shape: {tuple(z.shape)} dtype: {z.dtype} device: {z.device}")

		if until_stage == 2:
			return outputs

		# 阶段三：逐补丁语义特征提取
		stage3 = self._stage3_patch_semantics(outputs['proxy_image'], outputs['Z0_pre'])
		outputs.update(stage3)
		print("[Stage3] 完成：已生成并保存语义地图")
		# 复制语义地图到统一目录
		if 'semantic_map_path' in outputs and os.path.exists(outputs['semantic_map_path']):
			import shutil
			shutil.copy(outputs['semantic_map_path'], os.path.join(base_out_dir, f"semantic_map_{self._current_prompt_hash}.pt"))

		if until_stage == 3:
			return outputs

		# 阶段四：动态语义定位水印生成
		stage4 = self._stage4_generate_wlocs(outputs['semantic_vectors'], outputs['Z0_pre'])
		outputs.update(stage4)
		print("[Stage4] 完成：已生成并保存 W_loc^S")
		# 保存W_loc^S
		if 'W_loc_S_path' in outputs and os.path.exists(outputs['W_loc_S_path']):
			import shutil
			shutil.copy(outputs['W_loc_S_path'], os.path.join(base_out_dir, f"w_loc_s_{self._current_prompt_hash}.pt"))

		if until_stage == 4:
			return outputs

		# 阶段五：版权水印生成
		stage5 = self._stage5_generate_wcop()
		outputs.update(stage5)
		print("[Stage5] 完成：已生成并保存 W_cop")
		# 保存W_cop
		if 'W_cop_path' in outputs and os.path.exists(outputs['W_cop_path']):
			import shutil
			shutil.copy(outputs['W_cop_path'], os.path.join(base_out_dir, f"w_cop_{self._current_prompt_hash}.pt"))

		if until_stage == 5:
			return outputs

		# 阶段六：DMJS生成 Z_T^w
		stage6 = self._stage6_dmjs_and_prepare(outputs['W_cop'], outputs['W_loc_S'], outputs['Z0_pre'])
		outputs.update(stage6)
		print("[Stage6] 完成：已生成并保存 Z_T^w")
		# 保存Z_T^w
		if 'Z_T_w_path' in outputs and os.path.exists(outputs['Z_T_w_path']):
			import shutil
			shutil.copy(outputs['Z_T_w_path'], os.path.join(base_out_dir, f"z_tw_{self._current_prompt_hash}.pt"))

		# 阶段六内：用 Z_T^w 直接去噪生成最终图像
		stage6b = self._stage6b_generate_image(
			prompt,
			outputs['Z_T_w'],
			num_inference_steps=self.num_inference_steps,
			guidance_scale=self.guidance_scale,
		)
		outputs.update(stage6b)
		print("[Stage6] 完成：已生成最终图像")
		# 保存最终图像
		if 'final_image_path' in outputs and os.path.exists(outputs['final_image_path']):
			import shutil
			shutil.copy(outputs['final_image_path'], os.path.join(base_out_dir, f"image_{self._current_prompt_hash}.png"))

		return outputs


def build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='SEAL-LOC 单文件流水线（阶段式执行）')
	parser.add_argument('--prompt', type=str, required=True, help='文本提示词')
	parser.add_argument('--device', type=str, default='cuda', help='cuda 或 cpu')
	parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base', help='扩散模型ID或本地路径')
	parser.add_argument('--height', type=int, default=512)
	parser.add_argument('--width', type=int, default=512)
	parser.add_argument('--until_stage', type=int, default=6, help='执行到的阶段（1-6）')
	parser.add_argument('--patch_grid_size', type=int, default=8)
	parser.add_argument('--vlm_model_name', type=str, default='Salesforce/blip2-flan-t5-xl')
	parser.add_argument('--sentence_model_name', type=str, default='kasraarabi/finetuned-caption-embedding')
	parser.add_argument('--debug_stage3', action='store_true')
	parser.add_argument('--save_patch_grid_vis', action='store_true')
	parser.add_argument('--save_patch_crops', action='store_true')
	parser.add_argument('--log_patch_captions', action='store_true')
	parser.add_argument('--max_patch_crops', type=int, default=8)
	parser.add_argument('--simhash_bits', type=int, default=7)
	parser.add_argument('--wm_len', type=int, default=256)
	parser.add_argument('--tlt_intervals_num', type=int, default=3)
	parser.add_argument('--num_inference_steps', type=int, default=50)
	parser.add_argument('--guidance_scale', type=float, default=7.5)
	return parser


def main():
	parser = build_argparser()
	args = parser.parse_args()

	pipeline = SEALLOCPipeline(
		device=args.device,
		model_id=args.model_id,
		height=args.height,
		width=args.width,
		patch_grid_size=args.patch_grid_size,
		vlm_model_name=args.vlm_model_name,
		sentence_model_name=args.sentence_model_name,
		debug_stage3=args.debug_stage3,
		save_patch_grid_vis=args.save_patch_grid_vis,
		save_patch_crops=args.save_patch_crops,
		log_patch_captions=args.log_patch_captions,
		max_patch_crops=args.max_patch_crops,
		simhash_bits=args.simhash_bits,
		wm_len=args.wm_len,
		tlt_intervals_num=args.tlt_intervals_num,
		num_inference_steps=args.num_inference_steps,
		guidance_scale=args.guidance_scale,
	)
	outputs = pipeline.generate_loc_watermark(args.prompt, until_stage=args.until_stage)

	# 统一目录下信息提示
	print(f"[Output] 所有产物已统一保存至: {os.path.join(os.path.dirname(__file__), 'output', pipeline._current_prompt_hash)}")
	print("Pipeline finished up to stage:", args.until_stage)


if __name__ == '__main__':
	main() 