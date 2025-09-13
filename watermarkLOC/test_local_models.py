"""
测试本地模型加载

验证所有本地模型路径是否正确，模型是否能够正常加载
"""

import os
import torch
from model_config import get_model_path, check_model_exists

def test_model_paths():
    """测试模型路径是否存在"""
    print("🔍 检查本地模型路径...")
    
    models_to_check = [
        "stabilityai/stable-diffusion-2-1-base",
        "Salesforce/blip2-flan-t5-xl", 
        "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        "kasraarabi/finetuned-caption-embedding"
    ]
    
    for model_name in models_to_check:
        model_path = get_model_path(model_name)
        exists = check_model_exists(model_path)
        status = "✅" if exists else "❌"
        print(f"{status} {model_name}")
        print(f"   路径: {model_path}")
        if not exists:
            print(f"   错误: 路径不存在")
        print()

def test_diffusers_loading():
    """测试diffusers模型加载"""
    print("🔄 测试Stable Diffusion模型加载...")
    
    try:
        from diffusers import StableDiffusionPipeline
        
        model_path = get_model_path("stabilityai/stable-diffusion-2-1-base")
        print(f"📁 从路径加载: {model_path}")
        
        # 测试加载（不移动到GPU以节省内存）
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            device_map=None  # 不自动移动到设备
        )
        
        print("✅ Stable Diffusion模型加载成功")
        print(f"   模型类型: {type(pipe)}")
        
        # 清理内存
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"❌ Stable Diffusion模型加载失败: {e}")
        return False

def test_transformers_loading():
    """测试transformers模型加载"""
    print("🔄 测试BLIP-2模型加载...")
    
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        model_path = get_model_path("Salesforce/blip2-flan-t5-xl")
        print(f"📁 从路径加载: {model_path}")
        
        # 测试processor加载
        processor = Blip2Processor.from_pretrained(
            model_path,
            local_files_only=True
        )
        print("✅ BLIP-2 Processor加载成功")
        
        # 测试模型加载（不移动到GPU）
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            device_map=None
        )
        print("✅ BLIP-2 Model加载成功")
        print(f"   模型类型: {type(model)}")
        
        # 清理内存
        del processor, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"❌ BLIP-2模型加载失败: {e}")
        return False

def test_sentence_transformer_loading():
    """测试SentenceTransformer加载"""
    print("🔄 测试SentenceTransformer模型加载...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_path = get_model_path("kasraarabi/finetuned-caption-embedding")
        print(f"📁 从本地路径加载: {model_path}")
        
        # 尝试从本地路径加载
        model = SentenceTransformer(model_path)
        print("✅ SentenceTransformer本地加载成功")
        print(f"   模型类型: {type(model)}")
        
        # 简单测试
        test_text = "A beautiful landscape"
        embedding = model.encode(test_text)
        print(f"   测试编码维度: {embedding.shape}")
        
        # 清理内存
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"❌ SentenceTransformer本地加载失败: {e}")
        print("🔄 尝试在线加载...")
        try:
            model = SentenceTransformer("kasraarabi/finetuned-caption-embedding")
            print("✅ SentenceTransformer在线加载成功")
            
            # 简单测试
            test_text = "A beautiful landscape"
            embedding = model.encode(test_text)
            print(f"   测试编码维度: {embedding.shape}")
            
            # 清理内存
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return True
        except Exception as e2:
            print(f"❌ SentenceTransformer在线加载也失败: {e2}")
            return False

def main():
    """主测试函数"""
    print("🧪 SEAL-LOC 本地模型加载测试")
    print("=" * 50)
    
    # 1. 检查路径
    test_model_paths()
    
    # 2. 测试各个模型加载
    results = {}
    
    print("🔧 开始模型加载测试...")
    print("-" * 30)
    
    results['diffusers'] = test_diffusers_loading()
    print()
    
    results['transformers'] = test_transformers_loading() 
    print()
    
    results['sentence_transformer'] = test_sentence_transformer_loading()
    print()
    
    # 3. 总结结果
    print("📊 测试结果总结:")
    print("-" * 30)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！可以运行SEAL-LOC测试了")
        print("   建议运行: python simple_test.py")
    else:
        print("\n⚠️  部分测试失败，但可能仍可运行简化版本")
        print("   可以尝试: python simple_test.py --device cpu")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main()) 