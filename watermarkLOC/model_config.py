"""
模型配置文件
统一管理所有模型的本地路径配置
"""

import os

# 本地模型路径配置（根据用户提供的实际路径）
BLIP2_FLAN_T5_XL_PATH = "/home/wang003/.cache/modelscope/hub/models/zimuwangnlp/flan-t5-xl"
SD_2_1_BASE_PATH = "/home/wang003/.cache/modelscope/hub/models/AI-ModelScope/stable-diffusion-2-1-base"
CLIP_PATH = "/media/wang003/liyongqing/difusion/cache/models--laion--CLIP-ViT-g-14-laion2B-s12B-b42K/snapshots/4b0305adc6802b2632e11cbe6606a9bdd43d35c9"

# 语义编码模型路径（本地路径）
SENTENCE_TRANSFORMER_PATH = "/home/wang003/.cache/modelscope/hub/models/kasraarabi/finetuned-caption-embedding"

def get_model_path(model_name):
    """
    根据模型名称获取本地路径
    
    Args:
        model_name: 模型名称
        
    Returns:
        本地模型路径
    """
    model_paths = {
        "blip2-flan-t5-xl": BLIP2_FLAN_T5_XL_PATH,
        "Salesforce/blip2-flan-t5-xl": BLIP2_FLAN_T5_XL_PATH,
        "stable-diffusion-2-1-base": SD_2_1_BASE_PATH,
        "stabilityai/stable-diffusion-2-1-base": SD_2_1_BASE_PATH,
        "laion/CLIP-ViT-g-14-laion2B-s12B-b42K": CLIP_PATH,
        "sentence-transformer": SENTENCE_TRANSFORMER_PATH,
        "kasraarabi/finetuned-caption-embedding": SENTENCE_TRANSFORMER_PATH,
    }
    
    return model_paths.get(model_name, model_name)

def check_model_exists(model_path):
    """
    检查模型路径是否存在
    
    Args:
        model_path: 模型路径
        
    Returns:
        bool: 路径是否存在
    """
    if model_path.startswith("/"):  # 本地路径
        return os.path.exists(model_path)
    else:  # 在线模型
        return True  # 假设在线模型总是可用的 