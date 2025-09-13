# SEAL-LOC: 语义区域感知定位水印

基于语义的定位水印嵌入器，集成SEAL的语义感知能力与TAG-WM的双水印架构。

## 🎯 核心特性

- **语义感知**: 每个patch的水印模式与其语义内容密码学级别绑定
- **双水印架构**: 继承TAG-WM的版权水印(W_cop)和定位水印设计
- **动态生成**: 定位水印从固定模板(TLT)升级为动态语义绑定模板(W_loc^S)
- **完全兼容**: 保持与TAG-WM系统的完全兼容性

## 📁 项目结构

```
watermarkLOC/
├── seal_loc_embedder.py          # 主要嵌入器类
├── simhash_utils.py              # SimHash工具函数
├── patch_utils.py                # 补丁处理工具
├── test_seal_loc.py              # 测试脚本
├── semantic_maps/                # 语义地图存储目录
├── README.md                     # 项目文档
└── SEAL-LOC 水印嵌入模块设计文档.md  # 设计文档
```

## 🔧 安装依赖

```bash
# 基础依赖
pip install torch torchvision transformers sentence-transformers
pip install PIL numpy scipy

# 特定模型依赖
pip install diffusers accelerate
```

## 🚀 快速开始

### 基本使用

```python
from seal_loc_embedder import SEALLOCEmbedder
import torch

# 初始化SEAL-LOC嵌入器
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seal_loc = SEALLOCEmbedder(device=device)

# 准备参数
prompt = "A beautiful landscape with mountains and trees"
wm = torch.randint(0, 2, (256,), dtype=torch.float32, device=device)
latent_size = (4, 64, 64)

# 执行水印嵌入
latent_noise, w_cop, w_loc_s = seal_loc.embedding_seal_loc(
    prompt, wm, pipe, latent_size
)
```

### 语义地图管理

```python
# 保存语义地图
identifier = "unique_image_id"
seal_loc.save_semantic_map(semantic_vectors, identifier)

# 加载语义地图
loaded_vectors = seal_loc.load_semantic_map(identifier)
```

## 🧪 测试

运行完整测试套件：

```bash
cd watermarkLOC
python test_seal_loc.py --test all
```

运行特定测试：

```bash
# 语义提取测试
python test_seal_loc.py --test semantic

# 动态水印生成测试  
python test_seal_loc.py --test watermark

# DMJS集成测试
python test_seal_loc.py --test dmjs

# 端到端管线测试
python test_seal_loc.py --test e2e

# SimHash一致性测试
python test_seal_loc.py --test simhash

# Patch工具测试
python test_seal_loc.py --test patch
```

## 📊 性能参数

### 默认配置

- **网格大小**: 8×8 = 64个patch
- **语义向量维度**: 768维 (SentenceTransformer输出)
- **SimHash比特数**: 7位
- **采样策略**: 三区间DMJS
- **VLM模型**: Salesforce/blip2-flan-t5-xl
- **语义编码**: kasraarabi/finetuned-caption-embedding

### 可调参数

```python
seal_loc = SEALLOCEmbedder(
    patch_grid_size=8,              # 网格大小
    semantic_vector_dim=768,        # 语义向量维度
    simhash_bits=7,                 # SimHash比特数
    tlt_intervals_num=3,            # 采样区间数
    semantic_maps_dir='path/to/maps',  # 存储路径
    device='cuda'
)
```

## 🔄 工作流程

### 水印嵌入流程 (6个阶段)

1. **代理生成与初始表示**
   - 生成无水印的代理图像
   - 转换为潜空间表示

2. **逐补丁语义特征提取**
   - 8×8网格划分 (64个patch)
   - VLM生成每个patch的语义描述
   - 转换为768维语义向量

3. **动态语义定位水印生成**
   - 为每个patch独立计算SimHash
   - 生成确定性水印比特流
   - 拼接形成完整的W_loc^S

4. **版权水印生成**
   - 复用TAG-WM的W_cop生成逻辑
   - 重复展开并加密

5. **双水印联合采样 (DMJS)**
   - 三区间采样策略
   - (W_cop, W_loc^S)四种组合映射
   - 生成标准正态分布噪声

6. **扩散生成与解码**
   - 使用水印化噪声作为初始latents
   - 标准扩散去噪过程

## 🔍 关键算法

### SimHash算法

```python
def simhash_single_patch(embedding, num_bits=7, seed=42):
    bits = []
    for bit_index in range(num_bits):
        random_vector = torch.randn_like(embedding)
        dot_product = torch.dot(random_vector, embedding)
        bits.append(1 if dot_product > 0 else 0)
    
    return zlib.crc32(bytes(bits)) & 0xFFFFFFFF
```

### L2距离计算

```python
def calculate_patch_l2(noise1, noise2, k=64):
    # 分patch计算L2距离
    # 返回最小距离值
    return min_l2_distance
```

## 🎛️ 高级配置

### 自定义VLM模型

```python
seal_loc = SEALLOCEmbedder(
    vlm_model_name='your/custom-blip2-model',
    sentence_model_name='your/custom-sentence-transformer'
)
```

### 网格大小调整

```python
# 更精细的网格 (16×16 = 256个patch)
seal_loc = SEALLOCEmbedder(patch_grid_size=16)

# 更粗糙的网格 (4×4 = 16个patch)  
seal_loc = SEALLOCEmbedder(patch_grid_size=4)
```

## 📈 性能优化

### 批处理语义提取

```python
# 使用批量SimHash计算
from simhash_utils import simhash_batch_patches
hash_values = simhash_batch_patches(semantic_vectors)
```

### 语义地图缓存

```python
# 检查是否已存在语义地图
identifier = hashlib.md5(prompt.encode()).hexdigest()[:16]
cached_vectors = seal_loc.load_semantic_map(identifier)

if cached_vectors is None:
    # 重新计算语义向量
    semantic_vectors = seal_loc.extract_patch_semantics(image, latent_size)
    seal_loc.save_semantic_map(semantic_vectors, identifier)
else:
    semantic_vectors = cached_vectors
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```python
   # 使用CPU或减少batch size
   seal_loc = SEALLOCEmbedder(device='cpu')
   ```

2. **模型加载失败**
   ```python
   # 检查网络连接，或使用本地模型路径
   seal_loc = SEALLOCEmbedder(
       vlm_model_name='/path/to/local/blip2',
       sentence_model_name='/path/to/local/sentence-transformer'
   )
   ```

3. **SimHash不一致**
   ```python
   # 运行一致性测试
   from simhash_utils import verify_simhash_consistency
   is_consistent = verify_simhash_consistency(test_embedding)
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行单步测试
python test_seal_loc.py --test semantic
```

## 📚 参考资料

- [设计文档](./SEAL-LOC%20水印嵌入模块设计文档.md)
- [TAG-WM原始论文](../applied_to_sd2/)
- [SEAL原始实现](../baseline/)

## 🤝 贡献指南

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启Pull Request

## 📄 许可证

本项目基于原有TAG-WM和SEAL项目的许可证条款。

## 🎉 致谢

- TAG-WM团队提供的双水印架构基础
- SEAL团队提供的语义感知水印思路
- HuggingFace和OpenAI提供的预训练模型支持 