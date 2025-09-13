"""
独立的图像描述生成工具

从baseline/caption_pairs.py中提取generate_caption函数，
避免复杂的导入依赖问题。
"""

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from model_config import get_model_path


def generate_caption(image, processor, model, do_sample=False, device='cuda'):
    """
    使用BLIP-2模型为给定图像生成描述
    
    Args:
        image: PIL图像或图像路径
        processor: BLIP-2处理器
        model: BLIP-2模型
        do_sample: 是否使用采样生成
        device: 设备
        
    Returns:
        图像描述文本
    """
    # 处理输入图像
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise ValueError("Image must be either a file path or a PIL Image object")
    
    # 预处理图像
    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成描述
    with torch.no_grad():
        if do_sample:
            output = model.generate(**inputs, do_sample=True, top_p=0.9, max_length=50)
        else:
            output = model.generate(**inputs)
    
    # 解码输出
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def test_caption_generation():
    """测试图像描述生成功能"""
    print("Testing caption generation...")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        model_path = get_model_path('blip2-flan-t5-xl')
        processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            local_files_only=True
        ).to(device)
        
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color=(100, 150, 200))
        
        # 生成描述
        caption = generate_caption(test_image, processor, model, device=device)
        
        print(f"Generated caption: {caption}")
        print("Caption generation test passed!")
        
        return True
        
    except Exception as e:
        print(f"Caption generation test failed: {e}")
        return False


if __name__ == "__main__":
    test_caption_generation() 