# 论文错误报告 - 审稿前需修正的低级错误

## 1. 标点符号后缺少空格

### 第1页
- "data,offering" → 应改为 "data, offering" (摘要部分)
- "heterophily[2], [3]" → 应改为 "heterophily [2], [3]" (引言部分)
- "information1[4], [5]" → 应改为 "information [4], [5]" (脚注引用前应有空格)
- "Section 1." → 应改为 "Section I." (罗马数字一致性)

### 第2页
- "datasets,two" → 应改为 "datasets, two" (贡献部分)
- 多处参考文献引用的逗号后没有空格

### 第3页
- "x, y,z" → 空格不一致(应为 "x, y, z")

### 第4页
- **"datasetset"** → 应改为 "dataset" (严重拼写错误:重复单词 "heterophily datasetset")

### 第5页
- "GeomGCN." → 句点后跟引号,但与其他引用不一致

## 2. 引用和参考文献问题

### 引用括号前缺少空格
全文中引用括号前的空格不一致:
- 正确: "problem [1]"
- 错误: "problem[1]" 或 "problem [1],[2]"

**具体实例:**
- 第1页: "heterophily[2], [3]" 应为 "heterophily [2], [3]"
- 第2页: 多处引用分组缺少适当空格

### 引用格式不一致
- 有些引用在括号间有空格 [2], [3],而其他则分组 [2, 3]
- 需要在全文中保持一致的引用风格

## 3. 术语和大小写问题

### 第2页
- "Physical Review E's paper" → 应改为 "Physical Review E" (期刊名称应斜体或正确格式化)

### 第3页
- 章节标题大小写不一致("PRELIMINARIES" vs "Method")

## 4. 排版和格式问题

### 第1页
- "graph-diffusion model" 和 "graph diffusion model" - 全文中连字符使用不一致
- "over-smoothing" vs "oversmoothing" - 使用不一致(两种形式都出现)

### 第2页
- "Pois- ˇson" 在参考文献[12]中似乎有换行问题

### 第3页
- **图2中出现中文文本**: "背景知识 相关工作 研究方法 小论文思" - **这是严重错误** - 必须删除或翻译成英文

## 5. 数学符号问题

### 第3页
- 符号不一致: "X(t)" vs "x(t)" - 大写与小写使用不一致
- 需要验证这种区别是否是有意的(矩阵 vs 向量)

### 第4页
- "Xf (x)" 和 "Xh (x)" - 括号周围的空格应保持一致

## 6. 语法和措辞问题

### 第1页
- "This all stems from the fact that" → 冗长,可改为 "This stems from the fact that"
- "there is not a single study" → "there is no single study" (更自然)

### 第2页
- "they do not model it on a graph diffusion model for heterophily data" → 措辞笨拙,建议改为 "they do not develop a dedicated graph diffusion model for heterophily data"

### 第3页
- "named graph diffusion module and local information module" → 应为 "named the graph diffusion module and the local information module"

### 第4页
- **"heterophily datasetset"** → **严重拼写错误**: 重复单词

### 第5页
- "datasets derived from citation networks:" → 冒号后应在同一行跟随实际项目或使用新行格式

## 7. 图表问题

### 第2页
- 图1标题: "The nodes have distinct color with distinct label" → 应为 "The nodes have distinct colors with distinct labels" (复数)

### 第3页
- **图2包含中文文本** - 必须删除或翻译 - 这对英文论文投稿来说是严重错误

## 8. 补充材料引用

论文频繁提到"supplementary materials",但不清楚:
- 这些材料是否真实存在
- 是否正确引用
- 这对于投稿格式是否合适

**具体实例:**
- 第1页: 脚注引用补充材料
- 第2页: "Detailed examples can be found in the supplementary materials"
- 第3页: "For further details, please refer to the supplementary materials"

## 9. 术语不一致

- "heterophily" vs "heterophilic" - 交替使用,应验证每个上下文中的正确用法
- "graph neural networks" vs "graph neural network" - 单复数一致性
- "pseudo-labels" vs "pseudo labels" - 连字符不一致

## 10. 次要内容问题

### 第1页
- **"Physical Review.E"** → 应为 "Physical Review E" (删除E前的句点)

### 第5页
- 表III标题: "a-r r-e" → 数据集名称应更清晰地格式化或解释

### 第6页
- 参考文献[20]: "Spectral Graph Theory, p. 413–439, Jan 2015" → 这似乎是书中的一章,但缺少书名

## 11. 缩写问题

- "NIPS '21" vs "NIPS '19" - 某些参考文献中撇号使用不一致
- 缩写首次使用时应拼写完整(例如,"GNN"可能不是立即清楚的)

## 必须修复的严重错误总结

1. **图2中的中文文本** - 英文论文中完全不可接受
2. **第4页的"datasetset"拼写错误** - 重复单词
3. **全文逗号后缺少空格** - 普遍问题
4. **引用格式不一致** - 需要统一
5. **"Physical Review.E"** → 应为 "Physical Review E"

## 修改建议

1. 对整个文档运行拼写检查
2. 确保所有引用有一致的空格
3. 删除或翻译所有中文文本
4. 验证所有补充材料引用是否有效
5. 标准化连字符使用(为论文创建样式指南)
6. 检查所有图表引用是否实际存在
7. 校对重复单词
8. 确保每个逗号和句点后有空格(除行末外)

## 附加发现

### 语法细节
- 多处句子中 "the" 的使用不当或缺失
- 某些技术术语的冠词使用需要检查

### 一致性问题
- 确保全文中"heterophily dataset"的表述一致
- 检查所有模型名称的格式一致性(粗体、斜体或普通文本)

---

**优先级排序:**
1. 🔴 **最高优先级**: 删除图2中的中文文本
2. 🔴 **最高优先级**: 修正"datasetset"拼写错误
3. 🟡 **高优先级**: 修正"Physical Review.E"
4. 🟡 **高优先级**: 统一添加标点后的空格
5. 🟢 **中优先级**: 统一引用格式
6. 🟢 **中优先级**: 检查语法和措辞
