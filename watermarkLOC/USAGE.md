# SEAL-LOC 完整测试使用指南

## 🎯 概述

`complete_test.py` 是SEAL-LOC水印系统的完整测试文件，实现了以下完整流程：

1. **版权水印生成** - 使用TAG-WM方案生成版权水印Wcop
2. **语义定位水印生成** - 使用创新的语义感知方案生成定位水印Wloc  
3. **DMJS噪声生成** - 使用双水印联合采样生成初始噪声
4. **图像生成** - 利用Stable Diffusion 2.1生成水印化图像
5. **水印重建** - 通过DDIM反转重建水印并计算精度

## 🚀 快速开始

### 🧪 模型测试（建议先运行）

检查本地模型是否正确配置：

```bash
cd watermarkLOC
python test_local_models.py
```

### 🔧 简化测试（快速验证）

如果遇到导入问题，使用简化版本：

```bash
cd watermarkLOC
python simple_test.py
```

### 🔬 增强测试（高精度，推荐）

使用真正的DDIM反转，获得更准确的结果：

```bash
cd watermarkLOC
python enhanced_test.py
```

### 🎯 完整测试（功能完整）

```bash
cd watermarkLOC
python complete_test.py
```

### 自定义参数

```bash
# 简化测试
python simple_test.py \
    --prompt "A serene lake surrounded by mountains" \
    --device cuda \
    --output_dir output/my_test

# 完整测试  
python complete_test.py \
    --prompt "A serene lake surrounded by mountains" \
    --device cuda \
    --output_dir output/my_test \
    --num_tests 3
```

## 📋 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--prompt` | str | "A beautiful landscape..." | 测试提示词 |
| `--device` | str | cuda | 计算设备 (cuda/cpu) |
| `--model_id` | str | stabilityai/stable-diffusion-2-1-base | 扩散模型ID |
| `--output_dir` | str | output/complete_test | 输出目录 |
| `--num_tests` | int | 1 | 测试次数 |

## 📊 输出结果

测试完成后，会在输出目录生成以下文件：

```
output/complete_test/
├── test_1/
│   ├── watermarked_image.png      # 水印化图像
│   ├── watermark_data.npz         # 水印数据
│   ├── semantic_vectors.npz       # 语义向量
│   └── metrics.txt                # 详细指标
├── test_2/
│   └── ...
└── summary.txt                    # 平均结果汇总
```

## 📈 评估指标

测试系统会计算以下关键指标：

- **版权水印精度** - 版权水印比特重建准确率
- **定位水印精度** - 定位水印比特重建准确率  
- **L2距离** - patch级别的L2距离（SEAL评估标准）
- **执行时间** - 完整流程耗时

## 🔧 技术细节

### 版权水印 (Wcop)
- 长度：256位
- 编码：UTF-8字符串转二进制
- 加密：ChaCha20流加密
- 采样：DMJS三区间策略

### 语义定位水印 (Wloc)
- 网格：8×8 = 64个patch
- 语义模型：BLIP-2 + SentenceTransformer
- 哈希：SimHash (7位)
- 绑定：语义向量与水印模式密码学绑定

### 扩散模型
- 模型：Stable Diffusion 2.1 Base
- 精度：FP16半精度
- 潜空间：4×64×64
- 反转：DDIM 50步

## 🛠️ 故障排除

### 常见问题

1. **导入错误 (ModuleNotFoundError)**
   ```bash
   # 使用简化测试避免复杂依赖
   python simple_test.py --prompt "Your prompt"
   ```

2. **CUDA内存不足**
   ```bash
   python simple_test.py --device cpu
   # 或
   python complete_test.py --device cpu
   ```

3. **模型下载失败**
   - 检查网络连接
   - 使用代理或镜像
   - 预下载模型到本地

4. **依赖缺失**
   ```bash
   pip install -r ../requirements.txt
   pip install diffusers transformers torch torchvision
   ```

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 模拟测试

如果VLM模型加载失败，系统会自动回退到模拟语义向量，确保测试能够完成。

## 📝 示例输出

```
🚀 开始SEAL-LOC完整测试流程
📝 测试提示词: A beautiful landscape with mountains and trees
🔄 加载Stable Diffusion 2.1模型...
✅ 扩散模型加载成功
🔒 生成版权水印 (Wcop)...
✅ 版权水印生成完成 (长度: 256)
🎯 生成语义定位水印 (Wloc)...
✅ 语义定位水印生成完成 (形状: torch.Size([4, 64, 64]))
🎲 使用DMJS生成初始噪声...
✅ 初始噪声生成完成 (形状: torch.Size([1, 4, 64, 64]))
🎨 生成水印化图像...
✅ 水印化图像生成完成
🔍 重建水印...
📊 版权水印精度: 0.9453
📊 定位水印精度: 0.8721
📊 L2距离: 2.3456
✅ 完整测试流程完成！耗时: 45.67秒
```

## 🎯 性能优化

### GPU优化
- 使用FP16精度减少显存占用
- 批处理语义向量计算
- 缓存预训练模型

### 速度优化  
- 减少DDIM反转步数（trade-off精度）
- 使用更小的网格尺寸
- 预计算语义向量

## 🔬 实验配置

### 默认配置
```python
{
    "patch_grid_size": 8,           # 8×8网格
    "semantic_vector_dim": 768,     # 语义向量维度
    "simhash_bits": 7,              # SimHash位数
    "tlt_intervals_num": 3,         # DMJS区间数
    "wm_len": 256,                  # 水印长度
    "num_inference_steps": 50       # 推理步数
}
```

### 自定义配置
修改`complete_test.py`中的参数或创建配置文件。

## 📚 相关文档

- [设计文档](./SEAL-LOC%20水印嵌入模块设计文档.md)
- [项目README](./README.md)
- [TAG-WM原始代码](../applied_to_sd2/)
- [SEAL基线代码](../baseline/)

## 🤝 技术支持

如遇问题请检查：
1. 依赖安装是否完整
2. CUDA版本是否兼容
3. 模型文件是否正确下载
4. 内存/显存是否充足 