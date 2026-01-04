## Context

当前 ZeroG 项目使用编码器模型（BERT、RoBERTa、SentenceBERT）进行文本特征提取，这些模型通过 [CLS] token 获取句子表示。然而，最新的大型语言模型（如 Qwen、LLaMA 3、Mistral）采用解码器架构，在预训练时学习了更丰富的语言模式和知识。

### 背景
- **现状**: 仅支持双向编码器模型，使用第一个 token（[CLS]）作为句子表示
- **问题**: 无法利用强大的开源生成模型（Qwen、LLaMA 3）的能力
- **机遇**: 解码器模型可能在理解复杂数据集描述时表现更好，从而提升零样本迁移效果

### 技术约束
- 解码器模型通常更大（7B-70B 参数），需要更多 GPU 内存
- 因果语言模型使用最后一个 token 的表示而非第一个
- LoRA target_modules 因模型架构而异（Qwen 的 c_attn vs LLaMA 的 q_proj/v_proj）
- 需要正确处理 padding token（解码器模型通常没有专用的 pad_token）

### 利益相关者
- **研究人员**: 希望探索不同模型架构对零样本图学习的影响
- **用户**: 希望使用最新的开源模型提升性能
- **维护者**: 需要保持代码的清晰性和向后兼容性

## Goals / Non-Goals

### Goals
1. **支持主流解码器模型**: Qwen（通义千问）、LLaMA 3、Mistral
2. **正确的特征提取**: 使用最后一个 token 的隐藏状态作为句子表示
3. **参数高效微调**: 通过 LoRA 适配不同模型架构
4. **内存优化**: 支持 8-bit/4-bit 量化（QLoRA）以在消费级 GPU 上运行
5. **向后兼容**: 不破坏现有编码器模型的功能
6. **自动检测**: 自动识别模型类型并选择正确的特征提取策略

### Non-Goals
1. **全量微调**: 不支持对整个解码器模型的全参数微调（内存占用过大）
2. **模型压缩**: 不实现蒸馏、剪枝等模型压缩技术
3. **多语言扩展**: 初期只关注英文模型，不特别优化中文或其他语言
4. **生成任务**: 不使用解码器的生成能力，仅用于特征提取
5. **推理优化**: 不实现 vLLM、TensorRT 等专门的推理加速

## Decisions

### Decision 1: 特征提取策略

**选择**: 对于解码器模型，使用最后一个非填充 token 的隐藏状态作为句子表示。

**理由**:
- 解码器模型是因果的，最后一个 token 聚合了整个序列的信息
- 这是业界标准做法（如 LLaMA、GPT 的句子嵌入方法）
- 需要使用 attention_mask 定位实际的最后一个 token（排除填充）

**替代方案考虑**:
- ❌ **使用第一个 token**: 解码器的第一个 token 只看到自己，信息不完整
- ❌ **平均所有 token**: 会包含填充 token，引入噪声
- ❌ **使用专门的句子嵌入层**: 需要额外训练，增加复杂度

### Decision 2: LoRA Target Modules 配置

**选择**: 为每种模型架构维护专门的 target_modules 映射表。

**理由**:
- 不同模型的 attention 层命名差异很大（Qwen: c_attn, LLaMA: q_proj/v_proj）
- 错误的 target_modules 会导致 LoRA 无法应用或性能下降
- 显式映射比自动检测更可靠和可维护

**实现**:
```python
LORA_TARGET_MODULES = {
    'qwen': ['c_attn'],  # Qwen 使用融合的 attention 层
    'llama': ['q_proj', 'v_proj'],  # LLaMA 的 query 和 value 投影
    'llama3': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],  # LLaMA 3 扩展
    'mistral': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],  # Mistral 类似
}
```

**替代方案考虑**:
- ❌ **自动检测模块名称**: 容易出错，难以调试
- ❌ **统一命名**: 需要修改模型代码，不可行

### Decision 3: 内存优化策略

**选择**: 支持 QLoRA（4-bit 量化 + LoRA）作为主要内存优化方案。

**理由**:
- QLoRA 可以在 24GB GPU 上微调 70B 模型
- 性能损失很小（<1% 准确率下降）
- bitsandbytes 库已经成熟且广泛使用

**实现**:
```python
if args.load_in_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config
    )
```

**替代方案考虑**:
- ✅ **8-bit 量化**: 作为备选，内存节省少但更稳定
- ✅ **梯度检查点**: 可以叠加使用，进一步降低内存
- ❌ **模型并行**: 复杂度高，不适合单 GPU 场景

### Decision 4: 模型加载方式

**选择**: 支持两种加载方式：
1. Hugging Face Hub（在线下载）
2. 本地路径（离线使用）

**理由**:
- 灵活性：用户可以选择便捷的在线下载或离线部署
- 兼容性：兼容现有的 Hugging Face 生态
- 可重现性：本地路径确保固定版本

**实现**:
```python
if args.model_path:
    model_name_or_path = args.model_path
elif args.model_name:
    model_name_or_path = args.model_name  # e.g., "Qwen/Qwen2-7B"
else:
    model_name_or_path = DEFAULT_MODELS[args.text_encoder]
```

### Decision 5: Tokenizer Padding 配置

**选择**: 对于没有 pad_token 的模型，使用 eos_token 作为 pad_token。

**理由**:
- 许多解码器模型（如 LLaMA）没有专用的 pad_token
- 使用 eos_token 是社区标准做法
- 需要设置 `padding_side='left'` 以确保最后一个 token 是有效的

**实现**:
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'  # 左填充，右侧是实际内容
```

## Risks / Trade-offs

### 风险 1: 内存占用增加

**风险**: 解码器模型（7B-70B）比编码器模型（110M-340M）大 20-200 倍。

**缓解措施**:
- 强制要求使用 LoRA（不支持全量微调）
- 提供 QLoRA 选项（4-bit 量化）
- 在文档中明确说明 GPU 内存需求（7B 模型 + 4-bit 量化 ≈ 6GB）
- 提供梯度检查点选项进一步降低内存

**接受的权衡**: 训练速度可能降低 20-30%（量化和梯度检查点的开销）

### 风险 2: 推理速度下降

**风险**: 解码器模型的前向传播比编码器慢 3-10 倍。

**缓解措施**:
- 实现嵌入缓存（数据集描述和标签只编码一次）
- 使用批次推理
- 在文档中说明性能预期

**接受的权衡**: 零样本评估时间可能从几秒增加到几十秒

### 风险 3: 性能不确定性

**风险**: 解码器模型在图学习任务上的效果未经验证，可能不如编码器。

**缓解措施**:
- 保持编码器模型的完整支持（向后兼容）
- 在多个数据集上进行基准测试
- 提供性能对比文档

**接受的权衡**: 可能需要调整超参数（学习率、LoRA rank）以获得最佳性能

### 风险 4: 依赖版本冲突

**风险**: 新版本的 transformers、bitsandbytes 可能与现有依赖冲突。

**缓解措施**:
- 在 requirements.txt 中明确版本范围
- 提供 Docker 镜像以确保环境一致性
- 测试多种依赖版本组合

**接受的权衡**: 可能需要升级其他依赖（如 PyTorch）

## Migration Plan

### 阶段 1: 核心功能开发（预计 3-5 天）
1. 实现 `Text_Lora` 类的解码器支持
2. 添加特征提取逻辑切换
3. 配置 LoRA target_modules 映射
4. 更新命令行参数

### 阶段 2: 内存优化（预计 2-3 天）
1. 集成 bitsandbytes 量化
2. 实现梯度检查点
3. 添加嵌入缓存

### 阶段 3: 测试和验证（预计 3-4 天）
1. 在 Cora、Citeseer 数据集上测试 Qwen
2. 在 Arxiv 数据集上测试 LLaMA 3
3. 进行零样本迁移实验
4. 性能基准测试

### 阶段 4: 文档和示例（预计 1-2 天）
1. 更新 README.md
2. 创建示例脚本
3. 添加性能对比表格

### 回滚计划
- 所有新代码在 `if/else` 分支中，编码器路径不受影响
- 如果发现严重问题，可以简单地移除解码器选项
- 保留所有编码器模型的测试用例

## Open Questions

### 问题 1: 是否支持 Mixture of Experts (MoE) 模型？

**背景**: Qwen2-MoE、Mixtral 等 MoE 模型架构特殊。

**需要决定**:
- MoE 模型的 LoRA 应用策略（只对部分 expert 应用？）
- 内存和速度的权衡
- 是否值得增加额外的复杂度

**暂定方案**: 初期不支持，作为未来扩展

### 问题 2: 是否需要支持多模态模型？

**背景**: Qwen-VL、LLaVA 等模型支持图像输入。

**需要决定**:
- 图结构的可视化是否有助于节点分类
- 实现复杂度 vs 潜在收益

**暂定方案**: 不支持，超出当前变更范围

### 问题 3: 中文模型的特殊处理？

**背景**: Qwen、ChatGLM 等中文模型 tokenizer 可能有特殊行为。

**需要决定**:
- 是否需要针对中文数据集（如果有）进行特殊优化
- Tokenizer 配置的差异

**暂定方案**: 使用默认配置，遇到问题再针对性处理

### 问题 4: LoRA 超参数的默认值？

**背景**: 不同任务和模型可能需要不同的 LoRA rank、alpha。

**需要决定**:
- 默认 rank=4 是否适合解码器模型
- 是否需要模型特定的默认值

**暂定方案**:
- 默认 rank=8, alpha=16（比编码器略高）
- 提供命令行参数让用户自定义
