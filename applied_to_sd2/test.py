"""
端到端评测脚本（applied_to_sd2/test.py）

功能概述：
- 构建可反演的 Stable Diffusion 管线；
- 生成带版权水印/定位模板的图像；
- 对图像施加可选失真/篡改并生成真值掩码；
- 将图像编码并做 DDIM 正向推进，反解潜空间中的水印与定位模板；
- 计算水印与定位相关指标（以及可选的 CLIP 分数与噪声相似度）；
- 输出图像、掩码与 CSV 汇总文件。

使用说明（关键 CLI 参数示例）：
- --model_path, --scheduler：加载 SD 模型与调度器（DDIM/PNDM/UniPC/...）。
- --wm_len, --center_interval_ratio, --shuffle_random_seed, --encrypt_random_seed：水印设计相关。
- --tlt_intervals_num, --optimize_tamper_loc_method, --DVRD_checkpoint_path：定位细化相关。
- --img_h, --img_w, --num_inference_steps, --num_inversion_steps：生成与反演步数/尺寸。
- --logo_putting_num/random_crop_ratio 等：失真/篡改开关，--return_tamper_loc 返回真值掩码。
- --output_path：结果目录，自动创建 images、distorted_images、pred_tamper_loc_images 等子目录。
"""
import argparse
import copy
# from email.policy import default
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPTokenizer
from diffusers import DDIMScheduler, UniPCMultistepScheduler, PNDMScheduler, DEISMultistepScheduler, DPMSolverMultistepScheduler
import open_clip
import csv
from PIL import Image
import time
import os
from .inverse_stable_diffusion import InversableStableDiffusionPipeline
from .watermark_embedder import *
from utils.image_utils import *
from utils.optim_utils import *
from metrics.calc_imgs_similarity import *
from metrics.calc_tl_metrics import *
from metrics.calc_tpr_metrics import *

def main(args):
    """
    端到端主流程：
    1) 根据配置选择调度器与加载可反演 SD 管线；
    2) 构造水印/定位模板嵌入器 WatermarkEmbedder；
    3) 遍历数据集：生成图像 -> 施加失真/获取真值 -> 反演潜变量 -> 反解水印与定位模板；
    4) 统计水印准确率、定位指标、CLIP 分数与噪声相似度，并写入 CSV；
    5) 保存生成图、失真图、预测/真值掩码等产物。
    """
    device = args.device
    # Choose the scheduler
    if args.scheduler == 'DDIM':
        scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    elif args.scheduler == 'UniPC':
        scheduler = UniPCMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    elif args.scheduler == 'PNDM':
        scheduler = PNDMScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    elif args.scheduler == 'DEIS':
        scheduler = DEISMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    elif args.scheduler == 'DPMSolver':
        scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    # 1) 构造可反演的 SD 管线（不使用安全审查器），半精度推理以节省显存
    # Load pipe
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # 可选：加载参考 CLIP 模型用于图文一致性评估
    #reference model for CLIP Score
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model,
                                                                                  pretrained=args.reference_model_pretrain,
                                                                                  device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # 数据集与提示词入口（get_dataset 由外部 utils/数据集构造）
    # dataset
    dataset, prompt_key = get_dataset(args)

    # 2) 构造水印与定位模板嵌入器（支持 ChaCha20 加密、打乱、以及 DVRD 定位细化）
    # method of embedding watermark
    wm_tlt_embedder = WatermarkEmbedder(wm_len=args.wm_len,
                                        center_interval_ratio=args.center_interval_ratio,
                                        shuffle_random_seed=args.shuffle_random_seed,
                                        encrypt_random_seed=args.encrypt_random_seed,
                                        tlt_intervals_num=args.tlt_intervals_num,
                                        fpr=args.fpr, 
                                        user_number=args.user_number,
                                        optimize_tamper_loc_method=args.optimize_tamper_loc_method,
                                        DVRD_checkpoint_path=args.DVRD_checkpoint_path,
                                        DVRD_train_size=args.DVRD_train_size,
                                        device=device)

    # 结果目录与 CSV 汇总文件
    # create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # 检测阶段假设不知道原始 prompt（使用空串）
    # assume at the detection time, the original prompt is unknown
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # CSV 文件：基础指标、篡改定位、图像空间定位详细指标
    # Write infos to csv
    save_basic_info_path = os.path.join(args.output_path, "images_basic_infos.csv")
    # if not os.path.exists(save_basic_info_path):
    with open(save_basic_info_path, mode='w', newline='') as csv_file:
        csv_writer1 = csv.writer(csv_file)
        csv_writer1.writerow(["Image Name", "Image Path", "CLIP Score", 'Noise similarity', 'Watermark Accuracy', 'Watermark Global Accuracy', "Prompt", "Watermark", "Embedding Time Cost"])

    save_tamper_info_path = os.path.join(args.output_path, "images_tamper_infos.csv")
    # if not os.path.exists(save_tamper_info_path):
    with open(save_tamper_info_path, mode='w', newline='') as csv_file:
        csv_writer2 = csv.writer(csv_file)
        csv_writer2.writerow(["Image Name", "Image Path", \
            "Tamper Loc Acc(image)", 'Spatial Tamper Loc Acc(image)', 'Absolute Tamper Detection Error(image)', 'Calculated True Spatial Tamper Ratio(image)', \
            "Tamper Loc Acc(latent)", 'Spatial Tamper Loc Acc(latent)', 'Absolute Tamper Detection Error(latent)', 'Calculated True Spatial Tamper Ratio(latent)'])

    save_tamper_detailed_info_in_image_space_path = os.path.join(args.output_path, "infos_images_tamper_detailed_in_image_space.csv")
    # if not os.path.exists(save_tamper_detailed_info_in_image_space_path):
    with open(save_tamper_detailed_info_in_image_space_path, mode='w', newline='') as csv_file:
        csv_writer3 = csv.writer(csv_file)
        csv_writer3.writerow(["Image Name", 'Distortion Type', 'Accuracy', 'Precision', 'Specificity', 'Recall', 'AUC', 'IoU', 'Dice'])
            
    # 累积指标容器
    # wm_accs
    wm_accs = []
    wm_global_accs = []
    # CLIP Scores
    clip_scores = []
    # noise similarity
    noise_sims = []
    # for tamper localization
    latent_localize_accs = []
    latent_spatial_localize_accs = []
    latent_absolute_detect_errors = []
    latent_calc_true_spatial_tamper_ratios = []

    img_localize_accs = []
    img_spatial_localize_accs = []
    img_absolute_detect_errors = []
    img_calc_true_spatial_tamper_ratios = []

    embedding_time_costs = []
    gaussian_noise_time_costs = []
    sample_time_costs = []
    inversion_time_costs = []
    
    
    tl_metrics = {
        'Accuracy': 0,
        'Precision': 0,
        'Specificity': 0,
        'Recall': 0,
        'AUC': 0,
        'IoU': 0,
        'Dice': 0,
    }
    #test
    sample_idx = args.start_sample_idx
    # 3) 主循环：按数据项生成 -> 失真/真值 -> 反演 -> 反解 -> 统计与保存
    for i in tqdm(range(args.start_sample_idx, args.start_sample_idx + args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        # generate the watermark(wm) and the tamper localization template(tlt)
        set_random_seed(seed)

        # 生成随机比特水印；TLT 采用奇偶间隔（可替换为其他模板生成方式）
        wm = torch.randint(0, 2, [args.wm_len], requires_grad=False, device=device)
        # wm = torch.ones([args.wm_len], requires_grad=False, device=device).int()
        
        # center_num = int(args.latent_len * args.center_interval_ratio)
        # tlt = np.concatenate((np.zeros(center_num), np.ones(args.latent_len - center_num)), axis=0)
        tlt = np.arange(args.latent_len) % 2
        # tlt = np.random.randint(0, 2, args.latent_len)

        # embedding
        start_record_sample = 20
        if i >= start_record_sample:
            # 计时：嵌入 -> 采样噪声生成 -> 推理
            tic = time.time()
            init_latents_w, wm_repeat = wm_tlt_embedder.embedding_wm_tlt(wm, tlt, latent_size=args.latent_size)
            toc = time.time()
            embedding_time_cost = toc - tic
            print(f'embedding_wm_tlt time: {embedding_time_cost} seconds')
            embedding_time_costs.append(embedding_time_cost)
            
            tic = time.time()
            gaussian_noise = torch.randn(init_latents_w.shape, device=device)
            toc = time.time()
            gaussian_noise_time_cost = toc - tic
            print(f'gaussian_noise time: {gaussian_noise_time_cost} seconds')
            gaussian_noise_time_costs.append(gaussian_noise_time_cost)

            tic = time.time()
            outputs = pipe(
                current_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.img_h,
                width=args.img_w,
                latents=init_latents_w,
            )
            toc = time.time()
            sample_time_cost = toc - tic
            print(f'sample_time_cost time: {sample_time_cost} seconds')
            sample_time_costs.append(sample_time_cost)
        else:
            # 前若干张不计时，流程一致
            init_latents_w, wm_repeat = wm_tlt_embedder.embedding_wm_tlt(wm, tlt, latent_size=args.latent_size)
            outputs = pipe(
                current_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.img_h,
                width=args.img_w,
                latents=init_latents_w,
            )
            embedding_time_cost = gaussian_noise_time_cost = sample_time_cost = 'N/A'

        image_w = outputs.images[0]

        ### Save the generated image
        # 保存生成图像
        file_name = f"sample_{sample_idx:09d}.png"
        sample_idx += 1
        image_path = os.path.join(args.output_path, 'images', file_name)
        os.makedirs(os.path.join(args.output_path, 'images'), exist_ok=True)
        image_w.save(image_path)
        print(f"\tSaved image for prompt '{current_prompt}' at {image_path}")
        
        ## distortion
        if args.return_tamper_loc:
            # 对图像施加指定篡改与失真，并返回图像空间真值掩码
            image_w_distortion, true_tamper_loc_img = image_distortion(image_w, seed, args)
            pil_true_tamper_loc_img = Image.fromarray(true_tamper_loc_img * 255)
            true_tamper_loc_img = torch.from_numpy(true_tamper_loc_img).to(device).permute(2, 0, 1)
            true_tamper_loc_latent = wm_tlt_embedder.trans_tamper_loc_img2latent(pipe, true_tamper_loc_img, binaryize_thre=args.binaryize_thre)
            # save true tamper loc image
            true_tamper_loc_img_savedir = os.path.join(args.output_path, 'true_tamper_loc_images')
            os.makedirs(true_tamper_loc_img_savedir, exist_ok=True)
            true_tamper_loc_img_path = os.path.join(true_tamper_loc_img_savedir, file_name)
            pil_true_tamper_loc_img.save(true_tamper_loc_img_path)
                   
        else:
            # 仅失真，不返回真值掩码
            image_w_distortion = image_distortion(image_w, seed, args)
            true_tamper_loc_img = true_tamper_loc_latent = None
            
        # 保存失真图
        # save distorted image
        distorted_image_path = os.path.join(args.output_path, 'distorted_images', file_name)
        os.makedirs(os.path.join(args.output_path, 'distorted_images'), exist_ok=True)
        image_w_distortion.save(distorted_image_path)
         
        # reverse img
        # 4) 编码并正向 DDIM 推进获取反演潜变量
        image_w_distortion = transform_img(image_w_distortion, target_size=(args.img_h, args.img_w)).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
        if i >= start_record_sample:
            tic = time.time()
            reversed_latents_w = pipe.forward_diffusion(
                latents=image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=args.num_inversion_steps,
            )
            toc = time.time()
            inversion_time_cost = toc - tic
            print(f'inversion_time_cost time: {inversion_time_cost} seconds')
            inversion_time_costs.append(inversion_time_cost)
            
        else:
            reversed_latents_w = pipe.forward_diffusion(
                latents=image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=args.num_inversion_steps,
            )

        ### ----------------  deembedding ----------------  ###
        # 5) 反解水印与 TLT：反打乱 -> 反采样 -> 解密 -> 多数投票
        reversed_wm_repeat, reversed_tlt = wm_tlt_embedder.deembedding_wm_tlt(reversed_latents_w)
        pred_tamper_loc_latent = wm_tlt_embedder.get_tamper_loc_latent(tlt=tlt, 
                                                                       reversed_tlt=reversed_tlt, 
                                                                       latent_size=args.latent_size,
                                                                       tamper_confidence=args.tamper_confidence)
        if 'int' not in str(pred_tamper_loc_latent.dtype):
            pred_tamper_loc_latent_quantized = (torch.mean(pred_tamper_loc_latent, dim=0, keepdim=True) >= 0).int().repeat(4, 1, 1)
        else:
            pred_tamper_loc_latent_quantized = pred_tamper_loc_latent
        reversed_wm = wm_tlt_embedder.calc_watermark(wm_len=args.wm_len, 
                                                     wm_repeat=reversed_wm_repeat, 
                                                     pred_tamper_loc_latent=pred_tamper_loc_latent_quantized,
                                                     with_tamper_loc=args.calc_wm_use_tamper_loc)
        # reversed_wm = wm_tlt_embedder.calc_watermark(wm_len=args.wm_len, 
        #                                              wm_repeat=reversed_wm_repeat, 
        #                                              pred_tamper_loc_latent=pred_tamper_loc_latent,
        #                                              with_tamper_loc=args.calc_wm_use_tamper_loc)
        ### Calculate metrics
        # # Watermark acc
        # 水印位级准确率与全局重复匹配率
        wm_acc = wm_tlt_embedder.eval_watermark(wm=wm, reversed_wm=reversed_wm)
        wm_accs.append(wm_acc)
        wm_global_acc = (reversed_wm_repeat == wm_repeat).float().mean().item()
        wm_global_accs.append(wm_global_acc)

        # CLIP Score
        if args.reference_model is not None:
            socre = measure_similarity([image_w], current_prompt, ref_model,
                                              ref_clip_preprocess,
                                              ref_tokenizer, device)
            clip_socre = socre[0].item()
        else:
            clip_socre = 0
        clip_scores.append(clip_socre)

        # noise similarity metric
        # 初始嵌入噪声与反演噪声的余弦相似度，用于反演质量参考
        noise_sim = torch.cosine_similarity(init_latents_w[0].view(-1), reversed_latents_w[0].view(-1), dim=0).item()
        noise_sims.append(noise_sim)

        ##### ------------------------------------------- Tamper Localization ------------------------------------------- #####
        # 6) 将潜空间定位图解码到图像空间，用于可视化与指标计算
        # pred_tamper_loc_img = wm_tlt_embedder.trans_tamper_loc_latent2img(pipe, true_tamper_loc_latent, binaryize_thre=args.binaryize_thre) # test true
        pred_tamper_loc_img = wm_tlt_embedder.trans_tamper_loc_latent2img(pipe, pred_tamper_loc_latent, binaryize_thre=args.binaryize_thre)
        # pred_tamper_loc_img = wm_tlt_embedder.optimize_tamper_loc(pred_tamper_loc_img, tamper_confidence=args.tamper_confidence) # optimize
        pil_pred_tamper_loc_img = (pred_tamper_loc_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        # print('pil_pred_tamper_loc_img:', pil_pred_tamper_loc_img)
        pil_pred_tamper_loc_img = Image.fromarray(pil_pred_tamper_loc_img)
        
        # save pred tamper loc imgs
        # 保存预测定位掩码（图像空间）
        pred_tamper_loc_img_savedir = os.path.join(args.output_path, 'pred_tamper_loc_images')
        os.makedirs(pred_tamper_loc_img_savedir, exist_ok=True)
        pred_tamper_loc_img_path = os.path.join(pred_tamper_loc_img_savedir, file_name)
        pil_pred_tamper_loc_img.save(pred_tamper_loc_img_path)

        # Tamper Localization Acc
        # 7) 计算图像/潜空间的定位准确率与全局比例误差
        # in image space
        img_localize_acc, img_spatial_localize_acc, img_absolute_detect_error, img_calc_true_spatial_tamper_ratio = wm_tlt_embedder.eval_tamper_localization_and_detection(true_tamper_loc=true_tamper_loc_img, pred_tamper_loc=pred_tamper_loc_img)
        # in latent space
        latent_localize_acc, latent_spatial_localize_acc, latent_absolute_detect_error, latent_calc_true_spatial_tamper_ratio = wm_tlt_embedder.eval_tamper_localization_and_detection(true_tamper_loc=true_tamper_loc_latent, pred_tamper_loc=pred_tamper_loc_latent_quantized)
            
        img_localize_accs.append(img_localize_acc)
        img_spatial_localize_accs.append(img_spatial_localize_acc)
        img_absolute_detect_errors.append(img_absolute_detect_error)
        img_calc_true_spatial_tamper_ratios.append(img_calc_true_spatial_tamper_ratio)
        
        latent_localize_accs.append(latent_localize_acc)
        latent_spatial_localize_accs.append(latent_spatial_localize_acc)
        latent_absolute_detect_errors.append(latent_absolute_detect_error)
        latent_calc_true_spatial_tamper_ratios.append(latent_calc_true_spatial_tamper_ratio)

        # Tamper Localization Metrics
        # 8) 计算并累计图像空间定位的多项指标（Accuracy/Precision/Specificity/Recall/AUC/IoU/Dice）
        aggregated_metrics = calc_metrics(true_tamper_loc_img, pred_tamper_loc_img)
        tl_metrics['Accuracy'] += aggregated_metrics['Accuracy']
        tl_metrics['Precision'] += aggregated_metrics['Precision']
        tl_metrics['Specificity'] += aggregated_metrics['Specificity']
        tl_metrics['Recall'] += aggregated_metrics['Recall']
        tl_metrics['AUC'] += aggregated_metrics['AUC']
        tl_metrics['IoU'] += aggregated_metrics['IoU']
        tl_metrics['Dice'] += aggregated_metrics['Dice']
        ##### ------------------------------------------------------------------------------------------------------------------ #####

        # Write infos to csv
        with open(save_basic_info_path, mode='a', newline='') as csv_file:
            csv_writer1 = csv.writer(csv_file)
            csv_writer1.writerow([file_name, image_path, clip_socre, noise_sim, wm_acc, wm_global_acc, current_prompt, wm.flatten().cpu().numpy(), embedding_time_cost])

        with open(save_tamper_info_path, mode='a', newline='') as csv_file:
            csv_writer2 = csv.writer(csv_file)
            csv_writer2.writerow([file_name, image_path, \
                img_localize_acc, img_spatial_localize_acc, img_absolute_detect_error, img_calc_true_spatial_tamper_ratio, \
                latent_localize_acc, latent_spatial_localize_acc, latent_absolute_detect_error, latent_calc_true_spatial_tamper_ratio
                ])
        distortion_type = 'NoDistortion'
        if distortion_type == 'random_distortion':
            with open(save_tamper_detailed_info_in_image_space_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([file_name, distortion_suffix, aggregated_metrics['Accuracy'], aggregated_metrics['Precision'], aggregated_metrics['Specificity'], aggregated_metrics['Recall'], aggregated_metrics['AUC'], aggregated_metrics['IoU'], aggregated_metrics['Dice']])
        else:
            with open(save_tamper_detailed_info_in_image_space_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([file_name, distortion_type, aggregated_metrics['Accuracy'], aggregated_metrics['Precision'], aggregated_metrics['Specificity'], aggregated_metrics['Recall'], aggregated_metrics['AUC'], aggregated_metrics['IoU'], aggregated_metrics['Dice']])
    #tpr metric
    tpr_detection, tpr_traceability = wm_tlt_embedder.get_tpr()
    #save metrics
    save_metrics(args, tpr_detection, tpr_traceability, wm_accs, clip_scores)

    # save args and metrics
    basic_metrics = {
        'Start sample index': args.start_sample_idx,
        'Num of samples': args.num, 
        'Embedding Time Cost': sum(embedding_time_costs) / (args.num - start_record_sample),
        'Gaussian Noise Time Cost': sum(gaussian_noise_time_costs) / (args.num - start_record_sample),
        'Sample Time Cost': sum(sample_time_costs) / (args.num - start_record_sample),
        'Inversion Time Cost': sum(inversion_time_costs) / (args.num - start_record_sample),
        'CLIP scores': sum(clip_scores) / args.num,
        'Noise similarity': sum(noise_sims) / args.num,
        'Watermark detection Accuracy': sum(wm_accs) / args.num,
        'Watermark global detection Accuracy': sum(wm_global_accs) / args.num,
        'TPR_detection': tpr_detection / args.num,
        'TPR_traceability': tpr_traceability / args.num,
        }
    latent_tamper_loc_metrics = {
        'Tamper Localization Accuracy': sum(latent_localize_accs) / args.num,
        'Spatial Tamper Localization Accuracy': sum(latent_spatial_localize_accs) / args.num,
        'Tamper Absolute Detection Error': sum(latent_absolute_detect_errors) / args.num,
        'Calculated True Spatial Tamper Ratio': sum(latent_calc_true_spatial_tamper_ratios) / args.num,
        }
    image_tamper_loc_metrics = {
        'Tamper Localization Accuracy': sum(img_localize_accs) / args.num,
        'Spatial Tamper Localization Accuracy': sum(img_spatial_localize_accs) / args.num,
        'Tamper Absolute Detection Error': sum(img_absolute_detect_errors) / args.num,
        'Calculated True Spatial Tamper Ratio': sum(img_calc_true_spatial_tamper_ratios) / args.num,
        }
    image_detailed_tamper_loc_metrics = {
        'Accuracy': tl_metrics['Accuracy'] / args.num,
        'Precision': tl_metrics['Precision'] / args.num,
        'Specificity': tl_metrics['Specificity'] / args.num,
        'Recall': tl_metrics['Recall'] / args.num,
        'AUC': tl_metrics['AUC'] / args.num,
        'IoU': tl_metrics['IoU'] / args.num,
        'Dice': tl_metrics['Dice'] / args.num,
    }
    args_and_metrics = {
        'args': vars(args),
        'metrics': basic_metrics,
        'latent_tamper_loc_metrics': latent_tamper_loc_metrics,
        'image_tamper_loc_metrics': image_tamper_loc_metrics,
        'image_detailed_tamper_loc_metrics': image_detailed_tamper_loc_metrics,
        }
    with open(os.path.join(args.output_path, 'args_and_metrics.json'), 'w') as f:
        json.dump(args_and_metrics, f, indent=4)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    # basic settings
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gen_seed', default=0, type=int)

    # for watermark embedding 
    parser.add_argument('--wm_len', default=256, type=int)
    parser.add_argument('--wm_times', default=9, type=int)
    parser.add_argument('--adaptable_wm_times', default=True, type=bool)
    parser.add_argument('--wm_rough_ratio', default=0.3, type=float)
    parser.add_argument('--shuffle_random_seed', default=133563, type=int)
    parser.add_argument('--encrypt_random_seed', default=133563, type=int)
    parser.add_argument('--channel_copy', default=1, type=int, help=" (valid when watermark_embedding = 'dense') ")
    parser.add_argument('--hw_copy', default=8, type=int, help=" (valid when watermark_embedding = 'dense') ")
    parser.add_argument('--gen_template_from', default='fixed_bits', 
                        help=" 'watermark', 'fixed_bits' (valid when watermark_embedding = 'sparseWithTLT') ")

    # for generation and wm extraction
    parser.add_argument('--start_sample_idx', default=0, type=int)
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--scheduler', default='DPMSolver', help='DDIM | UniPC | PNDM | DEIS | DPMSolver')
    parser.add_argument('--img_h', default=512, type=int)
    parser.add_argument('--img_w', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=10, type=int)
    parser.add_argument('--dataset_path', default='./datasets/Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='/home/wang003/.cache/modelscope/hub/models/AI-ModelScope/stable-diffusion-2-1-base')
    parser.add_argument('--calc_wm_use_tamper_loc', default=False, type=bool)

    # for tamper localization
    parser.add_argument('--optimize_tamper_loc_method', default='trainable', type=str, help="'trainable' or 'trainfree'")
    parser.add_argument('--return_tamper_loc', default=False, type=bool)
    parser.add_argument('--binaryize_thre', default=0, type=float)
    parser.add_argument('--tamper_confidence', default=0.7, type=float)
    parser.add_argument('--center_interval_ratio', default=0.5, type=float)
    parser.add_argument('--tlt_intervals_num', default=3, type=float, help="when 'tlt_intervals_num'==4, center_interval_ratio must be 0.5")
    parser.add_argument('--DVRD_checkpoint_path', default="./DVRD/checkpoints/trainsize-512_epochnum-100_totalstep-33400.pt", type=str)
    parser.add_argument('--DVRD_train_size', default=512, type=int)

    # for content tamper
    parser.add_argument('--random_crop_ratio', default=None, type=float)    # random crop
    parser.add_argument('--random_drop_ratio', default=None, type=float)    # random drop
    parser.add_argument('--logo_putting_num', default=None, type=int)     # logo putting
    parser.add_argument('--logo_ratio', default=0.1, type=float)
    parser.add_argument('--logo_data_path', default='./datasets/SOIM/train/', type=str)

    # for spatial attacks
    parser.add_argument('--horizontal_shift_ratio', default=None, type=float, help="negative for up, positive for down")
    parser.add_argument('--vertical_shift_ratio', default=None, type=float, help="negative for left, positive for right")
                    
    # for image distortion
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--median_blur_k', default=None, type=int)
    parser.add_argument('--resize_ratio', default=None, type=float)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--sp_prob', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    
    # for model testing
    parser.add_argument('--reference_model', default='ViT-g-14')
    parser.add_argument('--reference_model_pretrain', default='laion2b_s12b_b42k')
    parser.add_argument('--output_path', default='./output/applied_to_sd2')
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_path, args.scheduler)
    
    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps
        
    args.latent_size = (4, args.img_h // 8, args.img_w // 8)
    args.latent_len = args.latent_size[0] * args.latent_size[1] * args.latent_size[2]

    if args.tlt_intervals_num == 4:
        args.center_interval_ratio = 0.5
    main(args)
