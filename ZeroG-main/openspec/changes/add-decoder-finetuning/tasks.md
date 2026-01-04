## 1. 模型架构实现

- [ ] 1.1 在 `st_model.py` 中添加解码器模型支持
  - [ ] 1.1.1 添加 Qwen 模型加载逻辑（AutoModelForCausalLM）
  - [ ] 1.1.2 添加 LLaMA 3 模型加载逻辑
  - [ ] 1.1.3 添加 Mistral 模型加载逻辑
  - [ ] 1.1.4 为每种解码器模型配置正确的 target_modules

- [ ] 1.2 实现特征提取策略切换
  - [ ] 1.2.1 添加 `is_decoder` 属性用于标识模型类型
  - [ ] 1.2.2 实现基于 attention_mask 的最后 token 位置定位
  - [ ] 1.2.3 修改 `forward()` 方法支持解码器特征提取
  - [ ] 1.2.4 修改 `zero_shot_eval()` 中的特征提取逻辑

- [ ] 1.3 扩展 LoRA 配置
  - [ ] 1.3.1 为 Qwen 配置 LoRA target_modules（["c_attn"] 或 ["q_proj", "v_proj"]）
  - [ ] 1.3.2 为 LLaMA 3 配置 LoRA target_modules（["q_proj", "v_proj", "k_proj", "o_proj"]）
  - [ ] 1.3.3 为 Mistral 配置 LoRA target_modules
  - [ ] 1.3.4 添加模型特定的 LoRA 超参数配置

## 2. 参数和配置

- [ ] 2.1 更新命令行参数（main_TextBP_benchmark.py）
  - [ ] 2.1.1 在 `--text_encoder` 中添加 'qwen', 'llama3', 'mistral' 选项
  - [ ] 2.1.2 添加 `--model_name` 参数用于指定 Hugging Face 模型标识符
  - [ ] 2.1.3 添加 `--model_path` 参数用于指定本地模型路径
  - [ ] 2.1.4 添加 `--lora_r` 和 `--lora_alpha` 参数用于配置 LoRA 超参数

- [ ] 2.2 添加内存优化选项
  - [ ] 2.2.1 添加 `--gradient_checkpointing` 标志
  - [ ] 2.2.2 添加 `--load_in_8bit` 标志
  - [ ] 2.2.3 添加 `--load_in_4bit` 标志（QLoRA）
  - [ ] 2.2.4 在模型初始化时应用这些优化

## 3. 工具函数

- [ ] 3.1 实现模型类型自动检测
  - [ ] 3.1.1 添加 `detect_model_type()` 函数检测编码器/解码器
  - [ ] 3.1.2 基于模型配置的 `architectures` 字段进行判断
  - [ ] 3.1.3 添加基于模型名称的回退检测逻辑

- [ ] 3.2 实现最后 token 位置提取
  - [ ] 3.2.1 添加 `get_last_token_position()` 函数
  - [ ] 3.2.2 使用 attention_mask 定位每个序列的最后一个非填充 token
  - [ ] 3.2.3 处理批次中的变长序列

## 4. 依赖更新

- [ ] 4.1 更新 requirements.txt
  - [ ] 4.1.1 确保 transformers >= 4.35.0（支持 Qwen、LLaMA 3）
  - [ ] 4.1.2 添加 bitsandbytes >= 0.41.0（8-bit/4-bit 量化）
  - [ ] 4.1.3 更新 PEFT >= 0.7.0（支持 QLoRA）
  - [ ] 4.1.4 添加 accelerate >= 0.24.0（模型加载优化）

## 5. 测试和验证

- [ ] 5.1 基础功能测试
  - [ ] 5.1.1 测试 Qwen 模型加载和特征提取
  - [ ] 5.1.2 测试 LLaMA 3 模型加载和特征提取
  - [ ] 5.1.3 测试 Mistral 模型加载和特征提取
  - [ ] 5.1.4 验证编码器模型的向后兼容性

- [ ] 5.2 LoRA 微调测试
  - [ ] 5.2.1 在 Cora 数据集上测试 Qwen + LoRA 微调
  - [ ] 5.2.2 在 Citeseer 数据集上测试 LLaMA 3 + LoRA 微调
  - [ ] 5.2.3 验证 LoRA 参数是否正确冻结/训练
  - [ ] 5.2.4 比较不同 LoRA rank 的性能差异

- [ ] 5.3 零样本评估测试
  - [ ] 5.3.1 在训练集（如 Cora）上训练，在测试集（如 Arxiv）上评估
  - [ ] 5.3.2 验证解码器模型的零样本迁移能力
  - [ ] 5.3.3 比较编码器和解码器模型的性能差异

- [ ] 5.4 内存优化测试
  - [ ] 5.4.1 测试 gradient checkpointing 的内存节省效果
  - [ ] 5.4.2 测试 8-bit 量化加载和微调
  - [ ] 5.4.3 测试 4-bit QLoRA 加载和微调
  - [ ] 5.4.4 记录不同配置下的 GPU 内存占用

## 6. 文档更新

- [ ] 6.1 更新 README.md
  - [ ] 6.1.1 添加解码器模型支持的说明
  - [ ] 6.1.2 提供使用示例（Qwen、LLaMA 3、Mistral）
  - [ ] 6.1.3 说明内存优化选项的使用方法
  - [ ] 6.1.4 更新依赖安装说明

- [ ] 6.2 更新代码注释
  - [ ] 6.2.1 在 `st_model.py` 中添加解码器相关的注释
  - [ ] 6.2.2 说明编码器和解码器的特征提取差异
  - [ ] 6.2.3 为新增参数添加清晰的 help 文本

- [ ] 6.3 添加使用示例脚本
  - [ ] 6.3.1 创建 `run_qwen.sh` 示例脚本
  - [ ] 6.3.2 创建 `run_llama3.sh` 示例脚本
  - [ ] 6.3.3 创建 `run_with_qlora.sh` 量化示例脚本

## 7. 性能优化

- [ ] 7.1 批次处理优化
  - [ ] 7.1.1 优化解码器模型的批次推理
  - [ ] 7.1.2 实现动态 padding 以减少计算浪费
  - [ ] 7.1.3 使用 torch.compile() 加速（PyTorch 2.0+）

- [ ] 7.2 缓存优化
  - [ ] 7.2.1 缓存数据集描述的文本嵌入
  - [ ] 7.2.2 缓存标签文本的嵌入
  - [ ] 7.2.3 避免重复的 tokenization 和编码
