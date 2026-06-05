# 面向嵌套命名实体识别的 GlobalPointer 创新方案

## 1. 研究重点

本文不再把重点放在医学领域知识增强、词典增强、显存压缩或推理加速上，而是聚焦一个更明确的问题：

> 如何在 GlobalPointer 基础上提升嵌套实体识别的 F1 分数。

当前代码 `CMeEE_GlobalPointer.py` 使用的是典型的 `BERT + GlobalPointer` 架构。它把命名实体识别转化为 span 分类问题：对每个实体类别、每个候选起止位置 `(start, end)` 计算一个分数，再通过阈值筛选得到实体。

这种方法天然适合嵌套实体，因为它允许多个 span 同时成立。例如：

```text
New York University
├── New York          LOC
└── New York University  ORG
```

但是，原始 GlobalPointer 仍然有一个核心不足：

> 它对每个候选 span 独立打分，缺少对 span 与 span 之间嵌套结构关系的显式建模。

因此，本文的创新应围绕“嵌套结构建模”展开，而不是围绕医学词典或通用工程优化展开。

## 2. 当前基线方法

当前代码的流程可以概括为：

1. 使用中文 BERT-base 编码文本。
2. 将实体标签转化为 `类别数 x maxlen x maxlen` 的 span 矩阵。
3. 使用 GlobalPointer 对每个候选 span 打分。
4. 使用多标签分类损失训练模型。
5. 预测时保留分数大于阈值 `0` 的 span。
6. 使用严格匹配计算 F1、Precision、Recall。

原方法的优势：

- 能统一处理嵌套实体和非嵌套实体。
- 避免传统 BIO 序列标注中一个 token 只能属于一个实体的问题。
- 结构简单，适合作为论文改进基线。

原方法的不足：

- 每个 span 独立判断，没有显式学习内层实体和外层实体之间的关系。
- 不合理的交叉 span 可能被同时预测出来。
- 固定阈值解码不能区分普通实体和嵌套实体。
- 对嵌套实体的 inner entity、outer entity 缺少专门优化。

## 3. 已有工作边界

为了避免把别人已经做过的内容包装成创新，下面这些方向不建议作为本文主创新。

| 方向 | 是否作为主创新 | 原因 |
|---|---|---|
| Efficient GlobalPointer | 否 | 已有方法，且已在 CMeEE 等数据集上使用 |
| 医学词典增强 | 否 | 偏医学领域，不符合当前“嵌套实体”主线 |
| 对抗训练 | 否 | 已有 GlobalPointer + adversarial training 工作 |
| 知识蒸馏 | 否 | 已有中文医学 NER + EfficientGlobalPointer + KD 工作 |
| 普通边界辅助任务 | 否 | boundary-enhanced / boundary-aware nested NER 已有较多工作 |
| 简单长度限制或 span mask | 否 | 已有 Dual-Masked Global Pointer、context window 等相近方法 |
| 重型 span graph / hypergraph / parser | 不作为主线 | 已有大量结构化 nested NER 工作，且会偏离当前代码基础 |

本文应避免的说法：

```text
本文提出 Efficient GlobalPointer。
本文提出医学词典增强 GlobalPointer。
本文提出边界辅助 GlobalPointer。
本文提出对抗训练提升 GlobalPointer。
本文提出知识蒸馏压缩 GlobalPointer。
本文通过长度限制减少候选 span。
```

更合适的定位是：

> 本文保留 GlobalPointer 的 span 矩阵建模优势，在其输出空间上加入轻量级嵌套结构关系建模、一致性训练和结构感知解码，以提升嵌套实体识别 F1。

## 4. 核心问题分析

### 4.1 独立 span 打分忽略嵌套关系

GlobalPointer 对候选 span 的打分形式可以理解为：

```text
score(c, i, j)
```

其中 `c` 是实体类别，`i` 是起点，`j` 是终点。

这个分数只表示“当前 span 是否为某类实体”，没有显式表达：

- 当前 span 是否包含另一个实体；
- 当前 span 是否被另一个实体包含；
- 两个 span 是否共享边界；
- 两个 span 是否发生不合理交叉；
- 内层实体是否能支持外层实体识别。

嵌套实体的关键不只是“识别一个 span”，而是“识别一组结构合理的 span”。

### 4.2 嵌套实体常见错误

在嵌套 NER 中，常见错误包括：

1. **内层实体漏检**：只识别外层实体，漏掉内部短实体。
2. **外层实体漏检**：只识别内部实体，没识别完整外层实体。
3. **边界偏移**：外层实体边界过长或过短。
4. **交叉误报**：预测出两个部分重叠但不是包含关系的 span。
5. **类别混淆**：内外层实体类型相互干扰。

因此，单纯提高普通实体 F1 不够，必须专门提升 nested-only F1。

## 5. 最终确定的创新点

### 创新点一：嵌套关系感知的 span 对建模

原始 GlobalPointer 只判断单个 span 是否成立。本文进一步在候选 span 之间建立关系判断。

对于两个候选 span：

```text
s_a = (c_a, i_a, j_a)
s_b = (c_b, i_b, j_b)
```

定义它们之间的结构关系：

```text
relation(s_a, s_b) ∈ {
  contain,
  inside,
  same-left,
  same-right,
  disjoint,
  crossing
}
```

含义：

- `contain`：`s_a` 包含 `s_b`。
- `inside`：`s_a` 被 `s_b` 包含。
- `same-left`：两个 span 共享左边界。
- `same-right`：两个 span 共享右边界。
- `disjoint`：两个 span 不相交。
- `crossing`：两个 span 部分重叠但不是包含关系。

创新点在于：

> 在 GlobalPointer 的候选 span 空间上显式建模 span-pair 嵌套关系，使模型不仅判断单个实体是否存在，还判断多个实体之间的结构是否合理。

### 创新点二：内外层实体一致性约束

嵌套实体不是一组互不相关的 span。内层实体和外层实体之间通常存在结构依赖。

例如：

```text
[New York]LOC University
[New York University]ORG
```

内层实体 `New York` 的存在可以帮助模型识别外层实体 `New York University`；外层实体的存在也说明内部 span 很可能不是噪声。

因此，训练时可以增加结构一致性损失：

```text
L_total = L_gp + α L_relation + β L_consistency
```

其中：

- `L_gp` 是原始 GlobalPointer 损失。
- `L_relation` 是 span-pair 关系分类损失。
- `L_consistency` 是内外层实体结构一致性损失。

一致性约束可以包括：

1. 包含关系中的内外层实体分数应互相支持。
2. 共享边界的合理 span 不应被互相压制。
3. 交叉重叠但非包含的 span 应受到惩罚。
4. 嵌套结构中的 outer span 和 inner span 应保持边界兼容。

创新点在于：

> 用结构一致性损失显式优化内外层实体协同识别，减少嵌套实体漏检和交叉误报。

### 创新点三：结构感知 GlobalPointer 解码

原始 GlobalPointer 解码方式是：

```text
score(c, i, j) > threshold
```

这会把所有候选 span 独立判断。本文将其改为结构感知打分：

```text
final_score(s)
  = gp_score(s)
  + λ1 * inner_outer_support(s)
  + λ2 * shared_boundary_support(s)
  - λ3 * crossing_penalty(s)
```

其中：

- `gp_score(s)` 是原始 GlobalPointer 分数。
- `inner_outer_support(s)` 表示内外层实体之间的支持。
- `shared_boundary_support(s)` 表示共享边界关系的支持。
- `crossing_penalty(s)` 惩罚结构不合理的交叉 span。

创新点在于：

> 解码时不再只看单个 span 分数，而是综合考虑候选实体集合的嵌套结构合理性。

## 6. 最终方法框架

整体框架可以写成：

```text
输入文本
  ↓
BERT 编码器
  ↓
GlobalPointer span 打分矩阵
  ↓
高分候选 span 选择
  ↓
span-pair 嵌套关系建模
  ↓
结构一致性训练
  ↓
结构感知解码
  ↓
实体输出
```

该方法不是重写 GlobalPointer，而是在它的基础上增加结构层：

```text
BERT + GlobalPointer + Nested Relation + Consistency Loss + Structure-aware Decoding
```

可以命名为：

```text
SA-GlobalPointer
Structure-aware GlobalPointer
```

中文名称：

```text
结构感知 GlobalPointer
```

## 7. 论文贡献写法

建议论文贡献写成三点：

1. 提出一种嵌套关系感知的 span 对建模方法，在 GlobalPointer 候选 span 空间中显式刻画包含、共享边界、相离和交叉等结构关系。
2. 设计内外层实体一致性约束，使模型在训练阶段学习嵌套实体之间的结构兼容性，减少内层实体和外层实体漏检。
3. 提出结构感知 GlobalPointer 解码策略，将内外层支持和交叉冲突惩罚注入 span 打分，提高嵌套实体识别 F1。

## 8. 实验设计

### 8.1 主实验

至少比较以下模型：

| 模型 | 作用 |
|---|---|
| BERT-CRF | 传统序列标注基线 |
| BERT-GlobalPointer | 当前代码基线 |
| BERT-EfficientGlobalPointer | 强 GlobalPointer 基线 |
| Boundary-enhanced span model | 边界增强类方法对比 |
| SA-GlobalPointer | 本文方法 |

### 8.2 消融实验

| 模型变体 | 验证目的 |
|---|---|
| GlobalPointer | 原始基线 |
| + span-pair relation | 验证嵌套关系建模 |
| + consistency loss | 验证结构一致性训练 |
| + structure-aware decoding | 验证结构解码 |
| full model | 完整模型效果 |

### 8.3 必须报告的指标

如果论文说重点是嵌套实体，就不能只报告整体 F1。建议增加：

1. **Overall F1**：整体严格匹配 F1。
2. **Nested-only F1**：只统计嵌套实体相关样本的 F1。
3. **Inner entity F1**：内层实体 F1。
4. **Outer entity F1**：外层实体 F1。
5. **Flat entity F1**：非嵌套实体 F1，防止方法损害普通实体。
6. **Crossing error rate**：交叉重叠误报率。
7. **Boundary error rate**：边界偏移错误率。

如果 `Nested-only F1`、`Inner F1`、`Outer F1` 上升，同时 `Crossing error rate` 下降，就能说明 F1 提升确实来自嵌套结构建模。

## 9. 实施路线

### 第一阶段：补充嵌套实体分析工具

先不要改模型，先统计数据集中的嵌套现象：

1. 多少样本包含嵌套实体。
2. 嵌套实体占总实体比例。
3. 最大嵌套深度。
4. inner entity 与 outer entity 的类型组合。
5. 共享左边界、共享右边界、完全包含的比例。

这一步可以支撑论文问题分析。

### 第二阶段：实现嵌套指标

在原有 F1 外增加：

- nested-only F1；
- inner F1；
- outer F1；
- crossing error rate；
- boundary error rate。

没有这些指标，后续很难证明方法确实改善嵌套实体。

### 第三阶段：实现 span-pair relation

从 GlobalPointer 预测矩阵中选取 top-k 或高分候选 span，对候选 span 对进行关系建模。

训练标签可由标注实体自动生成，不需要额外人工标注。

### 第四阶段：加入一致性损失

在原始 GlobalPointer loss 基础上加入：

```text
L_total = L_gp + α L_relation + β L_consistency
```

先做小权重实验，避免结构损失过强影响原始实体识别。

### 第五阶段：结构感知解码

在验证集上搜索：

- `λ1` inner/outer support 权重；
- `λ2` shared-boundary support 权重；
- `λ3` crossing penalty 权重；
- 原始阈值 threshold。

最终报告完整消融。

## 10. 推荐题目

中文题目：

> 面向嵌套命名实体识别的结构感知 GlobalPointer 方法研究

更强调 F1：

> 融合嵌套结构一致性的 GlobalPointer 命名实体识别方法

英文题目：

> Structure-aware GlobalPointer for Nested Named Entity Recognition

## 11. 参考文献线索

后续写论文时可重点围绕以下文献展开：

1. GlobalPointer 原始方法：<https://arxiv.org/abs/2208.03054>
2. Boundary Enhanced Neural Span Classification for Nested NER：<https://ojs.aaai.org/index.php/AAAI/article/view/6434>
3. Boundary-aware Neural Model for Nested NER：<https://aclanthology.org/D19-1034/>
4. Nested NER with Span-level Graphs：<https://aclanthology.org/2022.acl-long.63/>
5. Bottom-Up Constituency Parsing and Nested NER：<https://aclanthology.org/2022.acl-long.171/>

## 12. 当前代码实现映射

当前第一版代码实现优先选择对原始 `BERT + GlobalPointer` 侵入最小、最便于做消融的路线：

1. **span-pair 嵌套关系建模**：先落在候选 span 集合的结构分析与解码重打分阶段，不额外增加独立 relation classifier，避免训练图过重，也便于和原始 GlobalPointer 公平对比。
2. **内外层一致性约束**：在数据生成阶段自动识别参与嵌套关系的 gold span，并在损失函数中对这些 nested positive span 增加轻量正例一致性约束。
3. **结构感知解码**：在预测时综合 inner/outer support、shared-boundary support 与 crossing penalty 得到最终 span 分数。
4. **消融实验控制**：通过 `GP_ABLATION` 与独立环境变量开关控制模块启停，保证可以分别验证结构关系、一致性损失和结构解码对 F1 的贡献。

因此，当前代码对应的是轻量级 `SA-GlobalPointer` 第一版实现。后续如果实验结果显示结构解码有效，再考虑加入显式 relation classifier 或 pairwise relation loss，作为第二阶段增强，而不是在第一版中一次性引入过多变量。
