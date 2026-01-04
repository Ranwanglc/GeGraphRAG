## ADDED Requirements

### Requirement: 解码器模型架构支持

系统 SHALL 支持基于解码器架构的语言模型（如 Qwen、LLaMA、Mistral）进行 LoRA 微调，用于图节点的文本特征提取。

#### Scenario: 使用 Qwen 模型进行微调
- **WHEN** 用户指定 `--text_encoder qwen` 参数
- **THEN** 系统加载 Qwen 模型并配置适合该模型的 LoRA target_modules（如 "q_proj", "v_proj"）
- **AND** 使用最后一个 token 的隐藏状态作为文本嵌入
- **AND** 正确配置 tokenizer 的 padding token

#### Scenario: 使用 LLaMA 3 模型进行微调
- **WHEN** 用户指定 `--text_encoder llama3` 参数
- **THEN** 系统加载 LLaMA 3 模型并配置 LoRA 参数
- **AND** 使用因果语言模型的特征提取模式（最后一个 token）
- **AND** 设置 `tokenizer.pad_token = tokenizer.eos_token` 以处理填充

#### Scenario: 使用 Mistral 模型进行微调
- **WHEN** 用户指定 `--text_encoder mistral` 参数
- **THEN** 系统加载 Mistral 模型并配置相应的 LoRA 参数
- **AND** 正确处理解码器架构的隐藏状态提取

### Requirement: 解码器特征提取策略

系统 SHALL 根据模型类型（编码器/解码器）采用不同的文本特征提取策略。

#### Scenario: 编码器模型使用第一个 token
- **WHEN** 使用编码器模型（BERT、RoBERTa、SentenceBERT）
- **THEN** 从输出隐藏状态的第一个位置（[:,0,:]）提取特征
- **AND** 这对应于 [CLS] token 的表示

#### Scenario: 解码器模型使用最后一个 token
- **WHEN** 使用解码器模型（Qwen、LLaMA、Mistral）
- **THEN** 从输出隐藏状态的最后一个位置（[:,-1,:]）提取特征
- **AND** 使用 `output_hidden_states=True` 获取所有层的隐藏状态
- **AND** 从最后一层隐藏状态中提取特征

#### Scenario: 正确处理批次中的变长序列
- **WHEN** 处理不同长度的文本输入
- **THEN** 系统使用 attention_mask 识别每个序列的实际结束位置
- **AND** 对于解码器模型，从每个序列的最后一个非填充 token 位置提取特征

### Requirement: LoRA 配置适配

系统 SHALL 根据不同的解码器模型自动配置正确的 LoRA target_modules。

#### Scenario: Qwen 模型的 LoRA 配置
- **WHEN** 使用 Qwen 系列模型
- **THEN** target_modules 设置为 ["c_attn"] 或 ["q_proj", "v_proj", "k_proj"]
- **AND** LoRA rank 默认为 8，alpha 为 16
- **AND** lora_dropout 设置为 0.1

#### Scenario: LLaMA 模型的 LoRA 配置
- **WHEN** 使用 LLaMA 系列模型（LLaMA 2、LLaMA 3）
- **THEN** target_modules 设置为 ["q_proj", "v_proj"]
- **AND** 配置适合 LLaMA 架构的 LoRA 参数

#### Scenario: Mistral 模型的 LoRA 配置
- **WHEN** 使用 Mistral 模型
- **THEN** target_modules 设置为 ["q_proj", "v_proj", "k_proj", "o_proj"]
- **AND** 支持 sliding window attention 的特殊架构

### Requirement: 模型类型自动检测

系统 SHALL 能够自动检测模型是编码器还是解码器架构，并相应调整处理逻辑。

#### Scenario: 基于模型配置检测架构类型
- **WHEN** 加载预训练模型
- **THEN** 系统检查模型配置中的 `is_encoder_decoder` 或 `architectures` 字段
- **AND** 自动识别是否为因果语言模型（Causal LM）
- **AND** 设置内部标志以使用正确的特征提取策略

#### Scenario: 基于模型名称推断类型
- **WHEN** 模型配置不明确时
- **THEN** 系统根据模型名称关键词（qwen、llama、mistral、gpt）推断为解码器
- **AND** 根据名称关键词（bert、roberta、t5-encoder）推断为编码器

### Requirement: 命令行参数扩展

系统 SHALL 提供新的命令行参数以支持解码器模型的配置。

#### Scenario: 指定解码器模型路径
- **WHEN** 用户使用 `--text_encoder qwen --model_path /path/to/qwen`
- **THEN** 系统从指定路径加载 Qwen 模型
- **AND** 自动配置对应的 tokenizer 和 LoRA 参数

#### Scenario: 使用 Hugging Face 模型标识符
- **WHEN** 用户指定 `--text_encoder qwen --model_name Qwen/Qwen2-7B`
- **THEN** 系统从 Hugging Face Hub 下载并加载模型
- **AND** 缓存模型以供后续使用

#### Scenario: 配置 LoRA 超参数
- **WHEN** 用户指定 `--lora_r 16 --lora_alpha 32`
- **THEN** 系统使用指定的 LoRA rank 和 alpha 值
- **AND** 覆盖默认配置

### Requirement: 内存优化

系统 SHALL 提供内存优化选项以支持大型解码器模型的微调。

#### Scenario: 启用梯度检查点
- **WHEN** 用户启用 `--gradient_checkpointing` 标志
- **THEN** 系统在模型中启用梯度检查点以减少内存占用
- **AND** 训练速度可能略有下降但显著降低 GPU 内存需求

#### Scenario: 使用 8-bit 量化加载
- **WHEN** 用户指定 `--load_in_8bit` 标志
- **THEN** 系统使用 bitsandbytes 库以 8-bit 精度加载模型
- **AND** 大幅降低内存占用（约 50%）
- **AND** 保持微调性能

#### Scenario: 使用 4-bit 量化加载（QLoRA）
- **WHEN** 用户指定 `--load_in_4bit` 标志
- **THEN** 系统使用 QLoRA 方法以 4-bit 精度加载模型
- **AND** 内存占用降低约 75%
- **AND** 启用 nested quantization 和特殊的 LoRA 配置
