"""
Watermark and Tamper Localization Embedder (TAG-WM)

- 版权水印（bit 串）通过打乱（shuffle）、流加密（ChaCha20）后，在潜空间按“截断正态分布分段采样”规则联合模板 TLT 进行采样嵌入。
- TLT（Tamper Localization Template）采用对称/四分位阈值将 N(0,1) 分段，分别对应 (wm,tlt) 的四种组合，以保证可逆推与可定位性。
- 反向流程：解打乱 -> 反采样 -> 解密 -> 基于重复的多数投票恢复 wm；对比反推 TLT 与原始 TLT 得到潜空间篡改位置，再由 DVRD 细化，最后可映射到图像空间进行可视化与评估。
- 提供使用 DVRD（可训练/免训练）进行篡改定位细化的能力。
"""
import torch
import torch.nn as nn
from scipy.stats import norm, truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
from Crypto.Cipher import ChaCha20
# from Crypto.Random import get_random_bytes
import math
from bisect import bisect_left
from typing import Literal
from DVRD import api as DVRD_API
import time

class WatermarkEmbedder(torch.nn.Module):
    """
    密集版权水印 + 固定密集定位模板（TLT）嵌入器。

    设计要点：
    - 通过 `shuffle_random_seed` 实现固定可复现的打乱，提升鲁棒性与安全性。
    - 使用 ChaCha20 流加密对重复展开的水印序列进行加密，避免明文模式泄漏。
    - `tlt_intervals_num` 控制分段策略：3 表示以对称阈值 |z|>c 作为 tlt=1 区间；4 表示四分位切分（更细粒度）。
    - `optimize_tamper_loc_method` 支持 'trainable'（基于 DVRD UNet）与 'trainfree'（多尺度均值池化+二值化）。
    """
    def __init__(self, 
                 wm_len=256,
                 center_interval_ratio=0.5,
                 shuffle_random_seed=133563, 
                 encrypt_random_seed=133563, 
                 tlt_intervals_num=3,
                 fpr=1e-6, 
                 user_number=1000000,
                 optimize_tamper_loc_method: Literal['trainable', 'trainfree']=None,
                 DVRD_checkpoint_path=None,
                 DVRD_train_size=512,
                 device='cuda',
                 ):
        """
        参数：
        - wm_len: 水印比特长度（将在潜空间重复展开）
        - center_interval_ratio: 分段中心区间占比，用于确定截断阈值（tlt=0/1 的分界）
        - shuffle_random_seed: 打乱随机种子（None 则不打乱）
        - encrypt_random_seed: ChaCha20 加密种子（None 则不加密）
        - tlt_intervals_num: 3 或 4，控制截断分段策略
        - fpr, user_number: 计算一比特/多比特阈值 tau 用于 TPR 估计（误报率控制）
        - optimize_tamper_loc_method: 'trainable' | 'trainfree'，选择 DVRD 细化方式
        - DVRD_checkpoint_path, DVRD_train_size: DVRD 相关加载参数
        - device: 设备
        """
        super(WatermarkEmbedder, self).__init__()
        # basic settings
        self.device = device

        # settings for tamper localization template embedding
        self.center_interval_ratio = center_interval_ratio
        self.optimize_tamper_loc_method = optimize_tamper_loc_method
        if optimize_tamper_loc_method == 'trainable':
            self.TLR_module = DVRD_API.from_pretrained(checkpoint_path=DVRD_checkpoint_path, 
                                                      train_size=DVRD_train_size, 
                                                      torch_dtype=torch.float16, 
                                                      strict=False, 
                                                      device=device)

            print("Tamper Localization Refinement method: Trainable")
        elif optimize_tamper_loc_method == 'trainfree':
            self.TLR_module = DVRD_API.TrainfreeDVRD(max_kernel_size=None,
                                                          adaptive_max_kernel_size=True,
                                                          overlapping=False)
            print("Tamper Localization Refinement method: Train-free")

        self.tlt_intervals_num = tlt_intervals_num
        self.abs_truncdot = np.abs(norm.ppf(0.5 - self.center_interval_ratio / 2))
        
        # settings for watermark embedding
        self.wm_len = wm_len
        
        # settings for shuffle and de-shuffle
        self.shuffle_seed_generator = torch.Generator(device=device)
        self.shuffle_random_seed = shuffle_random_seed

        # settings for Chacha20 encrypt and decrypt
        self.encrypt_seed_generator = torch.Generator(device=device)
        self.encrypt_random_seed = encrypt_random_seed
        self.key = self.get_random_bytes(32)
        self.nonce = self.get_random_bytes(12)
        
        # settings for calculating tpr
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None
        for i in range(self.wm_len):
            fpr_onebit = betainc(i+1, self.wm_len - i, 0.5)
            fpr_bits = betainc(i+1, self.wm_len - i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.wm_len
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.wm_len

    def get_random_bytes(self, length):
        if self.encrypt_random_seed is None:
            return None
        self.encrypt_seed_generator.manual_seed(self.encrypt_random_seed) 
        random_ints = torch.randint(0, 256, (length,), dtype=torch.uint8, generator=self.encrypt_seed_generator, device=self.encrypt_seed_generator.device)
        return random_ints.cpu().numpy().tobytes()
    
    def stream_key_encrypt(self, sd):
        if self.encrypt_random_seed is None:
            return sd
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def stream_key_decrypt(self, reversed_m):
        if self.encrypt_random_seed is None:
            return reversed_m
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        return sd_bit
    
    def shuffle(self, data: torch.Tensor): # [data_length]
        if self.shuffle_random_seed is None:
            return data
        data_length = data.size(0)
        self.shuffle_seed_generator.manual_seed(self.shuffle_random_seed) 
        shuffle_indices = torch.randperm(data_length, generator=self.shuffle_seed_generator, device=data.device)
        # print('shuffle_indices:', shuffle_indices)
        shuffled_data = data[shuffle_indices]
        return shuffled_data

    def inverse_shuffle(self, shuffled_data: torch.Tensor): # [data_length]
        if self.shuffle_random_seed is None:
            return shuffled_data
        data_length = shuffled_data.size(0)
        self.shuffle_seed_generator.manual_seed(self.shuffle_random_seed) 
        shuffle_indices = torch.randperm(data_length, generator=self.shuffle_seed_generator, device=shuffled_data.device)
        # print('shuffle_indices:', shuffle_indices)
        inverse_indices = torch.empty_like(shuffle_indices, device=shuffled_data.device)
        inverse_indices[shuffle_indices] = torch.arange(data_length, device=shuffled_data.device)
        restored_data = shuffled_data[inverse_indices]
        return restored_data
    
    # def denseWMandDenseFixedTLTtruncSampling(self, wm, total_len):  # four intervals: (wm, tlt): (0, 1), (0, 0), (1, 0), (1, 1) 
    #     # Tamper Localization Template: [0, 0, 0, 0, 1, 1, 1,....]
    #     if total_len != self.latent_len:
    #         self.latent_len = total_len
    #         self.tlt = np.concatenate((np.zeros(total_len * self.center_interval_ratio)), np.ones(total_len *  (1 - self.center_interval_ratio)), axis=0)    #     z = np.zeros(total_len)
    #     self.truncdot = norm.ppf(0.5 - self.center_interval_ratio / 2)
    #     ppf = [-math.inf, self.truncdot, 0, -self.truncdot, math.inf]  # split 4 sampling range
    #     for i in range(0, total_len):
    #         if wm[i] == 0 and self.tlt[i] == 1:
    #             z[i] = truncnorm.rvs(ppf[0], ppf[1])
    #         elif wm[i] == 0 and i % 2 == 1:
    #             z[i] = truncnorm.rvs(ppf[1], ppf[2])
    #         elif wm[i] == 1 and i % 2 == 0:
    #             z[i] = truncnorm.rvs(ppf[2], ppf[3])
    #         elif wm[i] == 1 and i % 2 == 1:
    #             z[i] = truncnorm.rvs(ppf[3], ppf[4])    
    #     return torch.from_numpy(z).half().to(self.device)

    def denseWMandDenseFixedTLTtruncSampling(self, wm, tlt):  
        """
        基于截断正态分布的分段采样，将 (wm,tlt) 组合映射到 4 个截断区间上。

        - 当 tlt_intervals_num == 3：使用对称阈值 ±c，将 |z|>c 视为 tlt=1；z>0 视为 wm=1。
        - 当 tlt_intervals_num == 4：使用四分位阈值，将区间拆成四块，对应 (wm,tlt) 的四种情况。

        返回：
        - torch.HalfTensor，一维向量（潜变量展平后的噪声采样），稍后会被打乱与重塑形状。
        """
        if self.tlt_intervals_num == 3:
            z = np.zeros(wm.shape[0])
            ppf = [-math.inf, -self.abs_truncdot, 0, self.abs_truncdot, math.inf]  # split 4 sampling range

            # Sample wm and tlt noise using vectorized operations
            indices_wm_0_tlt_1 = (wm == 0) & (tlt == 1)
            indices_wm_0_tlt_0 = (wm == 0) & (tlt == 0)
            indices_wm_1_tlt_0 = (wm == 1) & (tlt == 0)
            indices_wm_1_tlt_1 = (wm == 1) & (tlt == 1)

            z[indices_wm_0_tlt_1] = truncnorm.rvs(ppf[0], ppf[1], size=np.sum(indices_wm_0_tlt_1))
            z[indices_wm_0_tlt_0] = truncnorm.rvs(ppf[1], ppf[2], size=np.sum(indices_wm_0_tlt_0))
            z[indices_wm_1_tlt_0] = truncnorm.rvs(ppf[2], ppf[3], size=np.sum(indices_wm_1_tlt_0))
            z[indices_wm_1_tlt_1] = truncnorm.rvs(ppf[3], ppf[4], size=np.sum(indices_wm_1_tlt_1))

        elif self.tlt_intervals_num == 4:
            z = np.zeros(wm.shape[0])
            self.abs_truncdot = np.abs(norm.ppf(0.25))
            ppf = [-math.inf, -self.abs_truncdot, 0, self.abs_truncdot, math.inf]  # split 4 sampling range
            indices_wm_0_tlt_0 = (wm == 0) & (tlt == 0)
            indices_wm_0_tlt_1 = (wm == 0) & (tlt == 1)
            indices_wm_1_tlt_0 = (wm == 1) & (tlt == 0)
            indices_wm_1_tlt_1 = (wm == 1) & (tlt == 1)

            z[indices_wm_0_tlt_0] = truncnorm.rvs(ppf[0], ppf[1], size=np.sum(indices_wm_0_tlt_0))
            z[indices_wm_0_tlt_1] = truncnorm.rvs(ppf[1], ppf[2], size=np.sum(indices_wm_0_tlt_1))
            z[indices_wm_1_tlt_0] = truncnorm.rvs(ppf[2], ppf[3], size=np.sum(indices_wm_1_tlt_0))
            z[indices_wm_1_tlt_1] = truncnorm.rvs(ppf[3], ppf[4], size=np.sum(indices_wm_1_tlt_1))

        return torch.from_numpy(z).half().to(self.device)

    def reverseTruncSampling(self, sampled_messege):
        """
        反采样：将采样得到的连续噪声还原为 (wm,tlt)。

        步骤：
        1) wm 通过 z>0 判定；
        2) tlt 在 3-intervals 模式通过 |z|>c 判定；在 4-intervals 模式通过区间条件判定。
        """
        # Tamper Localization Template: [0, 0, 0, 0, 1, 1, 1,....]
        if isinstance(sampled_messege, torch.Tensor):
            sampled_messege = sampled_messege.detach().cpu().numpy() 
            
        # reverse wm 
        reversed_wm = (sampled_messege > 0).astype(int)

        # reverse tlt
        if self.tlt_intervals_num == 3:
            reversed_tlt = (np.abs(sampled_messege) > self.abs_truncdot).astype(int)
            
        elif self.tlt_intervals_num == 4:
            # Define the conditions for each interval
            conditions = [
                (sampled_messege < -self.abs_truncdot),  # Condition for the first interval
                (sampled_messege >= -self.abs_truncdot) & (sampled_messege < 0),  # Condition for the second interval
                (sampled_messege >= 0) & (sampled_messege < self.abs_truncdot),  # Condition for the third interval
                (sampled_messege >= self.abs_truncdot)  # Condition for the fourth interval
            ]
            # Define the corresponding values for each condition
            choices = [0, 1, 0, 1]
            # Use np.select() to apply the conditions in batch
            reversed_tlt = np.select(conditions, choices, default=None)  # 'default' is used for cases that don't satisfy any condition
            # Check if any condition was met, if not, raise an error
            if reversed_tlt is None or np.any(reversed_tlt is None):
                raise ValueError("An unexpected value encountered. No condition met.")
        return reversed_wm, reversed_tlt    

    def embedding_wm_tlt(self, wm, tlt, latent_size):
        """
        将 wm 与 tlt 编码到潜变量噪声：
        - 重复展开 wm 以覆盖潜空间长度；
        - 流加密（可选）保护比特序列；
        - 基于 (wm,tlt) 执行分段截断采样得到一维噪声；
        - 打乱并重塑为 (1,C,H,W) 的潜变量形状。

        返回：latent_noise, wm_repeat（torch.Tensor）
        """
        # calculate latent length
        wm_len = wm.shape[0]
        latent_len = tlt.shape[0]
        # repeat watermark
        wm_times = int(latent_len // wm_len)
        remain_wm_len = latent_len % wm_len
        wm_repeat = torch.concat([wm.repeat(wm_times), wm[:remain_wm_len]], dim=0)
        # encrypt
        wm_repeat_encrypt = self.stream_key_encrypt(wm_repeat.cpu().numpy())
        # sampling
        flat_latent = self.denseWMandDenseFixedTLTtruncSampling(wm_repeat_encrypt, tlt)
        # shuffle
        shuffled_flat_latent = self.shuffle(flat_latent)
        # reshape
        latent_noise = shuffled_flat_latent.reshape(1, *latent_size)
        return latent_noise, wm_repeat

    def deembedding_wm_tlt(self, reversed_latent: torch.Tensor):
        """
        反解码：
        - 展平 -> 反打乱 -> 反采样得到 (wm_repeat_encrypt, reversed_tlt) -> 解密得到 wm_repeat
        返回：wm_repeat (float Tensor 0/1)、reversed_tlt (np.ndarray 0/1)
        """
        # flaten
        flat_reversed_latent = reversed_latent.view(-1)
        # de-shuffle
        deshuffled_flat_reversed_latent = self.inverse_shuffle(flat_reversed_latent)
        # de-sampling
        wm_repeat_encrypt, reversed_tlt = self.reverseTruncSampling(deshuffled_flat_reversed_latent)
        # decrypt
        wm_repeat = self.stream_key_decrypt(wm_repeat_encrypt)    
        wm_repeat = torch.from_numpy(wm_repeat).to(reversed_latent.device).float()
        return wm_repeat, reversed_tlt

    # def calc_watermark(self, wm_len, wm_repeat, pred_tamper_loc_latent=None, with_tamper_loc=True):
    #     if 'int' not in str(pred_tamper_loc_latent.dtype):
    #         pred_tamper_loc_latent = (pred_tamper_loc_latent - torch.min(pred_tamper_loc_latent)) / (torch.max(pred_tamper_loc_latent) - torch.min(pred_tamper_loc_latent))
    #     latent_len = wm_repeat.size(0)
    #     wm_repeat_times = latent_len // wm_len
    #     complete_wm_len = wm_len * wm_repeat_times
    #     remain_wm_len = latent_len - complete_wm_len

    #     if with_tamper_loc == False or pred_tamper_loc_latent is None:
    #         pred_tamper_loc_latent = torch.zeros_like(wm_repeat)

    #     # 将篡改位置转换为权重（非篡改区域的置信度）
    #     pred_notamper_weight = 1 - pred_tamper_loc_latent.view(-1)  # 保持浮点型
    #     pred_notamper_weight = self.inverse_shuffle(pred_notamper_weight)
        
    #     # 加权水印信号（非篡改区域的贡献度）
    #     wm_repeat_weighted = wm_repeat * pred_notamper_weight

    #     # 分块处理（保持浮点计算）
    #     # 处理完整区块
    #     split_weighted_wm = torch.cat(
    #         torch.split(wm_repeat_weighted[:complete_wm_len].unsqueeze(0), wm_len, dim=1),
    #         dim=0
    #     )
    #     split_weights = torch.cat(
    #         torch.split(pred_notamper_weight[:complete_wm_len].unsqueeze(0), wm_len, dim=1),
    #         dim=0
    #     )

    #     # 处理剩余部分
    #     if remain_wm_len > 0:
    #         remaining_weighted_wm = torch.cat([
    #             wm_repeat_weighted[complete_wm_len:],
    #             torch.zeros(wm_len - remain_wm_len, device=wm_repeat.device)
    #         ]).unsqueeze(0)
            
    #         remaining_weights = torch.cat([
    #             pred_notamper_weight[complete_wm_len:],
    #             torch.zeros(wm_len - remain_wm_len, device=wm_repeat.device)
    #         ]).unsqueeze(0)
            
    #         split_weighted_wm = torch.cat([split_weighted_wm, remaining_weighted_wm], dim=0)
    #         split_weights = torch.cat([split_weights, remaining_weights], dim=0)

    #     # 加权投票（防止除零）
    #     total_weighted_vote = torch.sum(split_weighted_wm, dim=0)  # 加权投票总和
    #     total_weights = torch.sum(split_weights, dim=0)            # 总权重
        
    #     # 添加极小值防止除零
    #     epsilon = 1e-8
    #     reversed_watermark = (total_weighted_vote / (total_weights + epsilon) >= 0.5).int()

    #     return reversed_watermark

    def calc_watermark(self, wm_len, wm_repeat, pred_tamper_loc_latent=None, with_tamper_loc=True):
        """
        基于重复展开的 wm_repeat 恢复原始 wm：
        - 将 DVRD/模板差异得到的“非篡改置信度”(1 - tamper) 反打乱后作为加权；
        - 按 wm_len 分块进行加权投票，得到最终 bit 结果；
        - 当 with_tamper_loc=False 或无定位信息时，退化为等权投票。
        返回：reversed_watermark (int Tensor 0/1)
        """
        latent_len = wm_repeat.size(0)   
        wm_repeat_times = latent_len // wm_len
        complete_wm_len = wm_len * wm_repeat_times
        remain_wm_len = latent_len - complete_wm_len
        
        if with_tamper_loc == False or pred_tamper_loc_latent is None:
            pred_tamper_loc_latent = torch.zeros_like(wm_repeat)
            
        pred_notamper_loc_latent = 1 - pred_tamper_loc_latent.view(-1)
        pred_notamper_loc_latent = self.inverse_shuffle(pred_notamper_loc_latent)
        wm_repeat_notamper = wm_repeat * pred_notamper_loc_latent

        # de-repeat
        split_notamper_wm = torch.cat(torch.split(wm_repeat_notamper[:complete_wm_len].unsqueeze(0), wm_len, dim=1), dim=0)
        remaining_notamper_wm = torch.cat([wm_repeat_notamper[complete_wm_len:], torch.zeros(wm_len - remain_wm_len, device=wm_repeat.device)]).unsqueeze(0)
        split_notamper_wm = torch.cat([split_notamper_wm, remaining_notamper_wm], dim=0)

        split_notamper = torch.cat(torch.split(pred_notamper_loc_latent[:complete_wm_len].unsqueeze(0), wm_len, dim=1), dim=0)
        remaining_notamper = torch.cat([pred_notamper_loc_latent[complete_wm_len:], torch.zeros(wm_len - remain_wm_len, device=wm_repeat.device)]).unsqueeze(0)
        split_notamper = torch.cat([split_notamper, remaining_notamper], dim=0)
        
        # watermark vote
        vote = torch.sum(split_notamper_wm, dim=0)
        # print('vote:', vote)
        vote_num = torch.sum(split_notamper, dim=0)
        # print('vote_num:', vote_num)
        reversed_watermark = (vote / vote_num >= 0.5).int()
        # print('reversed_watermark:', reversed_watermark)
            
        return reversed_watermark


    def eval_watermark(self, wm, reversed_wm):
        """
        评估版权水印比特级准确率，并基于设定的阈值累计 TPR 统计。
        返回：accuracy（float）
        """
        acc = (reversed_wm == wm).float().mean().item()
        if acc >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if acc >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        print(f"Copyright Watermark Accuracy: {acc: .4f}")
        return acc
    
    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count
    
    ### Following Codes for Tamper Localization 
    def get_tamper_loc_latent(self, tlt, reversed_tlt, latent_size, tamper_confidence=0.5, optimize=True, return_initial_tl=False):
        """
        基于模板差异 (reversed_tlt != tlt) 得到初始潜空间篡改位置图，并可选择使用 DVRD 进行细化：
        - trainable：UNet 结构（4 通道输入/输出），对潜空间 4 通道进行细化；
        - trainfree：多尺度均值池化聚合，随后二值化，复制到 4 通道。
        返回：细化后/原始的潜空间篡改图 (C=4,H,W)
        """
        tamper_loc_latent = (reversed_tlt != tlt).astype(int)
        tamper_loc_latent = torch.from_numpy(tamper_loc_latent).to(self.device)
        tamper_loc_latent = self.shuffle(tamper_loc_latent)
        tamper_loc_latent = tamper_loc_latent.reshape(*latent_size)
        if optimize:
            optimize_time_costs = []
            # optimize pred_tamper_loc
            if self.tlt_intervals_num == 3:
                tic = time.time()
                refined_tamper_loc_latent = self.optimize_tamper_loc(tamper_loc_latent, tamper_confidence=tamper_confidence, method=self.optimize_tamper_loc_method)
                toc = time.time()
                optimize_time_cost = toc - tic
                print(f'optimize time: {optimize_time_cost} seconds')
                optimize_time_costs.append(optimize_time_cost)
                print(f'optimize time cost (avg): {np.mean(optimize_time_costs)} seconds')
            elif self.tlt_intervals_num == 4:
                refined_tamper_loc_latent = self.optimize_tamper_loc(tamper_loc_latent, tamper_confidence=tamper_confidence, method=self.optimize_tamper_loc_method)
        else:
            refined_tamper_loc_latent = torch.mean(tamper_loc_latent.float(), dim=0, keepdim=True).repeat(4, 1, 1)
        if return_initial_tl:
            return refined_tamper_loc_latent, tamper_loc_latent
        else:
            return refined_tamper_loc_latent

    def optimize_tamper_loc(self, tamper_loc_latent, tamper_confidence=0.5, method: Literal['TLR', 'HFCD'] = 'TLR'):
        """
        篡改定位细化：
        - 'trainable'：调用 DVRD UNet 前向，半精度输入/输出；
        - 'trainfree'：通道均值 -> DVRD.TrainfreeDVRD 进行多尺度聚合与阈值化 -> 复制到 4 通道。
        """
        if method == 'trainable':
            return self.TLR_module(tamper_loc_latent.half().unsqueeze(0))[0]
            # return self.TLR_module(tamper_loc_latent.float().unsqueeze(0))[0]
        elif method == 'trainfree':
            tamper_loc_latent = torch.mean(tamper_loc_latent.float(), dim=0, keepdim=True) # (work)
            tamper_loc_latent = self.TLR_module(tamper_loc_latent, confidence=tamper_confidence)
            tamper_loc_latent = tamper_loc_latent.squeeze(0).repeat(4, 1, 1).int()  # (work)  # try
            return tamper_loc_latent

    def trans_tamper_loc_img2latent(self, pipe, tamper_loc_img, binaryize_thre=0, value_continuous=False):
        """
        将图像空间的篡改掩码映射到潜空间：
        - 使用 VAE 编码（不采样）得到潜变量；
        - 可选地对潜变量二值化并复制到 4 通道。
        """
        if tamper_loc_img.dim() == 3:
            tamper_loc_img = tamper_loc_img.unsqueeze(0)
        tamper_loc_img = tamper_loc_img.half()
        tamper_loc_latent = pipe.get_image_latents(tamper_loc_img * 2 - 1, sample=False)[0]
        
        if value_continuous is False:
            tamper_loc_latent = torch.mean(tamper_loc_latent.float(), dim=0, keepdim=True).repeat(4, 1, 1)
            tamper_loc_latent = (tamper_loc_latent >= binaryize_thre).int()
            
        return tamper_loc_latent
    
    def trans_tamper_loc_latent2img(self, pipe, tamper_loc_latent, binaryize_thre=0, value_continuous=True):
        """
        将潜空间篡改图解码到图像空间，便于可视化与度量。
        - 若 value_continuous=False，则先做值域映射与二值化。
        返回：C=3 的图像掩码（0/1）
        """
        if tamper_loc_latent.dim() == 3:
            tamper_loc_latent = tamper_loc_latent.unsqueeze(0)
            
        if value_continuous is False:
            tamper_loc_latents = tamper_loc_latents * 2 - 1
            
        tamper_loc_latent = tamper_loc_latent.half()
        tamper_loc_img = pipe.decode_image(tamper_loc_latent)[0]
        tamper_loc_img = torch.mean(tamper_loc_img, dim=0).repeat(3, 1, 1)
        tamper_loc_img = (tamper_loc_img >= binaryize_thre).int()
        return tamper_loc_img


    def calc_spatial_tamper_loc(self, tamper_local_loc):
        return (np.mean(tamper_local_loc, axis=0) >= 0.5).astype(int)

    def eval_tamper_localization_and_detection(self, pred_tamper_loc, true_tamper_loc=None):
        """
        评估定位水印性能：
        - 图像/潜空间逐像素准确率（localize_acc）与空间聚合后的准确率（spatial_localize_acc）。
        - 同时给出全局篡改比例的绝对误差与真实比例。
        返回：(localize_acc, spatial_localize_acc, absolute_detect_error, calc_true_spatial_tamper_ratio)
        """
        """
        tamper loc value:
            1. True or 1 -- tampered
            2. False or 0 -- not tampered
        """
        # transform data type
        if isinstance(pred_tamper_loc, torch.Tensor):
            pred_tamper_loc = pred_tamper_loc.detach().cpu().numpy()
        if isinstance(true_tamper_loc, torch.Tensor):
            true_tamper_loc = true_tamper_loc.detach().cpu().numpy()
            
        # generate predictied tamper and calculate predicted tamper ratio
        pred_spatial_tamper_loc = self.calc_spatial_tamper_loc(pred_tamper_loc)
        pred_spatial_tamper_ratio = np.mean(pred_spatial_tamper_loc)
        # print
        # print('Pred tamper loc:', pred_tamper_loc)
        # print('Pred spatial tamper loc:', pred_spatial_tamper_loc)

        # generate true tamper and calculate true tamper ratio
        if true_tamper_loc is None:
            true_tamper_loc = np.zeros_like(pred_tamper_loc)
            
        true_spatial_loc_tamper = self.calc_spatial_tamper_loc(true_tamper_loc)
        calc_true_spatial_tamper_ratio = np.mean(true_spatial_loc_tamper)

        # calculate detection accuracy
        localize_acc = np.mean(pred_tamper_loc == true_tamper_loc)
        spatial_localize_acc = np.mean(pred_spatial_tamper_loc == true_spatial_loc_tamper)
        absolute_detect_error = np.abs(calc_true_spatial_tamper_ratio - pred_spatial_tamper_ratio)

        # print
        print('Tamper Localization Acc:', localize_acc) 
        print('Spatial Tamper Localization Acc:', spatial_localize_acc) 
        print('Absolute Detection Error:', absolute_detect_error) 
        print('Calculated True Spatial Tamper Ratio:', calc_true_spatial_tamper_ratio) 

        return localize_acc, spatial_localize_acc, absolute_detect_error, calc_true_spatial_tamper_ratio
    


        
if __name__ == '__main__':
    # Sparse Watermark Embedder
    batch_size = 1
    wm_length = 256
    latent_size = (4, 64, 64) 
    device = 'cuda'
    watermark_embedder = DenseWMandDenseFixedTLTEmbedder(device=device)
    init_latents_w = watermark_embedder.watermark_embedding(latent_size=latent_size, requires_grad=False)
    reversed_latents_w = init_latents_w
    acc_metric = watermark_embedder.eval_watermark(reversed_latents_w)
    print('acc:', acc_metric)

    # Position Watermark Embedder
