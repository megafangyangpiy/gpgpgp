# 本地变更记录

时间戳：2026-06-05 09:50:44 +08:00

仓库地址：https://github.com/megafangyangpiy/gpgpgp.git

本地工作目录：`D:\A\gp\GlobalPointer`

## Kaggle 复现相关改动

- 修改 `CMeEE_GlobalPointer.py`，适配 Kaggle 的只读 input 和可写 working 目录规则。
- 默认从 `/kaggle/input/datasets/megafangyangpiy/gpgpgpgpgp` 读取项目输入。
- 模型权重和生成文件只写入 `/kaggle/working`。
- 新增环境变量覆盖入口：
  - `GP_INPUT_DIR`：覆盖 Kaggle input 根目录。
  - `GP_OUTPUT_DIR`：覆盖输出目录，默认 `/kaggle/working`。
  - `GP_EVAL_BATCH_SIZE`：覆盖验证阶段批量预测大小，默认 `16`。

## 依赖兼容相关改动

- 在导入 `bert4keras` 前加入 legacy TensorFlow/Keras 开关：
  - `TF_KERAS=1`
  - `TF_USE_LEGACY_KERAS=1`
- 加入 NumPy 兼容补丁，用于绕过 Kaggle 当前 TensorFlow 导入时的 `_no_nep50_warning` 报错。
- 将 `bert4keras.optimizers.Adam` 替换为后端 `keras.optimizers.Adam`，避免新版 Keras 中 `_set_hyper` 不存在的问题。
- 新增 `requirements_kaggle.txt`，记录 Kaggle 依赖安装方式。

## 日志与验证速度相关改动

- 将 `model.predict(..., verbose=0)` 设为静默预测，避免验证阶段反复输出 `1/1 ...`。
- 通过统一的 `TQDM_KWARGS` 降低进度条刷新频率，减少 Kaggle 输出刷屏。
- 将 CMeEE 验证从逐条预测改成批量预测。
- 默认验证批大小为 `GP_EVAL_BATCH_SIZE=16`。
- 用户在 Kaggle P100 上观察到批量验证前 GPU 占用约 `8.3 GB / 16 GB`。
- 调参建议：
  - 如果显存稳定且想进一步加速，可以尝试 `GP_EVAL_BATCH_SIZE=32`。
  - 如果验证阶段出现 GPU OOM，降低到 `GP_EVAL_BATCH_SIZE=8`。

## GitHub 推送记录

- `8c616b9`：首次提交 Kaggle GlobalPointer 项目。
- `3fc06a4`：减少 Kaggle 验证阶段日志刷屏。
- `db2822b`：批量化 CMeEE 验证，并记录 Kaggle GPU 使用情况。

## 当前 Kaggle 运行命令

```bash
python /kaggle/input/datasets/megafangyangpiy/gpgpgpgpgp/CMeEE_GlobalPointer.py
```

## 2026-06-05 12:35:48 +08:00

### 本次新增改动

- 开始实现 Sparse GlobalPointer 原型，先在当前 `CMeEE_GlobalPointer.py` 内完成，不重写底层 `GlobalPointer` 层。
- 新增稀疏化相关环境变量：
  - `GP_SPARSE_MAX_SPAN_LEN`：最大候选 span 长度，默认 `128`。
  - `GP_SPARSE_TOPK`：预测阶段每条样本保留的候选 span 数量，默认 `512`。
  - `GP_SPARSE_LOSS_MASK`：是否在训练损失中启用 sparse span mask，默认启用。
- 训练阶段新增 sparse span mask：
  - 去掉 `end < start` 的非法 span。
  - 去掉超过最大长度的 span。
  - 训练期强制保留 gold span，避免真实实体被稀疏 mask 误删。
- 预测阶段新增 sparse decode：
  - 先过滤非法和过长 span。
  - 再按模型分数保留 top-k 候选。
  - 为后续 span graph 提供更少、更干净的候选节点。

### 设计说明

- 这是第一版 Sparse GlobalPointer 原型，目标是先验证“候选稀疏化”和“召回保持”是否可行。
- 当前版本主要减少 loss/decoding 参与的 span 候选和验证输出候选数；还没有重写 GlobalPointer 内部矩阵计算。
- 后续如果该版本验证稳定，再继续做结构反哺式 span 图。

## 2026-06-05 12:54:13 +08:00

### 本次新增改动

- 为 Sparse GlobalPointer 做 1 epoch 对照实验，将 `CMeEE_GlobalPointer.py` 中的训练阶段 sparse loss mask 直接关闭：
  - 原逻辑：`sparse_loss_mask = os.environ.get('GP_SPARSE_LOSS_MASK', '1') != '0'`
  - 当前逻辑：`sparse_loss_mask = False`
- 当前设置用于验证“训练阶段 sparse loss mask 是否带来收益”。
- 注意：预测阶段 sparse decode 仍然保留，包括最大 span 长度过滤和 top-k 候选保留。

### 对照实验目的

- 和上一轮默认开启 sparse loss mask 的 1 epoch 结果对比。
- 重点观察：
  - `valid f1`
  - `precision`
  - `recall`
  - 训练时间和显存是否变化

## 2026-06-05 13:07:57 +08:00

### Sparse loss mask 对照实验结果

- 开启 sparse loss mask 的 1 epoch 结果：
  - `loss = 0.8968`
  - `global_pointer_f1_score = 0.4581`
  - `valid f1 = 0.62948`
  - `precision = 0.59538`
  - `recall = 0.66773`
  - 单 epoch 时间约 `416s`
- 关闭 sparse loss mask 的 1 epoch 结果：
  - `loss = 0.9039`
  - `global_pointer_f1_score = 0.4555`
  - `valid f1 = 0.60200`
  - `precision = 0.67401`
  - `recall = 0.54389`
  - 单 epoch 时间约 `400s`

### 初步结论

- 关闭 sparse loss mask 后 precision 上升，但 recall 明显下降。
- `valid f1` 下降约 `0.02748`，recall 下降约 `0.12384`。
- 对 CMeEE 这类实体识别任务，召回下降风险更关键，因此当前更建议默认开启 sparse loss mask。
- 训练时间差异不明显，用户本轮未记录显存变化。

### 本次代码状态

- 已将 `CMeEE_GlobalPointer.py` 恢复为默认开启 sparse loss mask：
  - `sparse_loss_mask = os.environ.get('GP_SPARSE_LOSS_MASK', '1') != '0'`
- 后续仍可通过 `GP_SPARSE_LOSS_MASK=0` 关闭该机制做临时对照。

## 2026-06-05 13:21:09 +08:00

### `GP_SPARSE_MAX_SPAN_LEN=256` 单 epoch 结果

- 实验设置：
  - `GP_SPARSE_MAX_SPAN_LEN=256`
  - 其余 sparse 设置保持当前默认逻辑。
- 1 epoch 结果：
  - `loss = 0.9192`
  - `global_pointer_f1_score = 0.4490`
  - `valid f1 = 0.58982`
  - `precision = 0.71166`
  - `recall = 0.50360`
  - 单 epoch 时间约 `397s`

### 初步观察

- 相比 `GP_SPARSE_MAX_SPAN_LEN=128` 且开启 sparse loss mask 的结果：
  - `valid f1` 从 `0.62948` 降到 `0.58982`。
  - `recall` 从 `0.66773` 降到 `0.50360`。
  - `precision` 从 `0.59538` 升到 `0.71166`。
- 该结果表现为 precision 提高但 recall 明显下降，暂时不建议把 `256` 作为默认配置。
- 当前更建议继续保留 `GP_SPARSE_MAX_SPAN_LEN=128` 做完整 10 epoch 实验。

## 2026-06-05 13:28:53 +08:00

### 本次新增改动

- 在 `CMeEE_GlobalPointer.py` 前部新增统一实验模式控制 `GP_EXPERIMENT_MODE`。
- 当前支持两种模式：
  - `GP_EXPERIMENT_MODE=1`：原版 GlobalPointer。
  - `GP_EXPERIMENT_MODE=2`：Sparse GlobalPointer。
- 默认模式设为 `2`，即继续使用 Sparse GlobalPointer。
- 当模式为 `1` 时：
  - 关闭训练阶段 sparse loss mask。
  - 关闭预测阶段最大 span 长度过滤。
  - 关闭预测阶段 top-k 候选裁剪。
- 当模式为 `2` 时：
  - 使用 `GP_SPARSE_MAX_SPAN_LEN` 控制最大候选 span 长度，默认 `128`。
  - 使用 `GP_SPARSE_TOPK` 控制预测候选数量，默认 `512`。
  - 使用 `GP_SPARSE_LOSS_MASK` 控制训练阶段 sparse loss mask，默认开启。

### 输出文件命名调整

- 不同实验模式的最佳权重文件分开保存，避免消融实验互相覆盖：
  - 原版：`best_model_cmeee_original_globalpointer.weights`
  - Sparse 版：`best_model_cmeee_sparse_globalpointer.weights`
- 预测输出文件也按模式区分：
  - 原版：`CMeEE_test_original_globalpointer.json`
  - Sparse 版：`CMeEE_test_sparse_globalpointer.json`

### Kaggle 使用方式

- 跑原版 GlobalPointer：

```python
import os
os.environ['GP_EXPERIMENT_MODE'] = '1'
```

- 跑 Sparse GlobalPointer：

```python
import os
os.environ['GP_EXPERIMENT_MODE'] = '2'
```

- 然后运行：

```bash
python /kaggle/input/datasets/megafangyangpiy/gpgpgpgpgp/CMeEE_GlobalPointer.py
```

## 2026-06-05 14:29:45 +08:00

### 原版与 Sparse GlobalPointer 10 epoch 对比结果

- 原版 GlobalPointer，`GP_EXPERIMENT_MODE=1`：
  - 最佳验证结果出现在 epoch 4。
  - `best valid f1 = 0.65665`
  - 对应 `precision = 0.66922`
  - 对应 `recall = 0.64453`
  - 第 1 个 epoch 时间约 `387s`
  - 后续 epoch 时间约 `310s - 319s`
- Sparse GlobalPointer，`GP_EXPERIMENT_MODE=2`，`GP_SPARSE_MAX_SPAN_LEN=128`，`GP_SPARSE_TOPK=512`，开启 sparse loss mask：
  - 最佳验证结果出现在 epoch 5。
  - `best valid f1 = 0.65023`
  - 对应 `precision = 0.64948`
  - 对应 `recall = 0.65099`
  - 第 1 个 epoch 时间约 `424s`
  - 后续 epoch 时间约 `326s - 335s`

### 实验结论

- 当前 Sparse GlobalPointer 原型没有带来预期收益：
  - 没有提升验证集 F1。
  - 没有减少训练时间。
  - 用户观察到显存占用也没有明显下降。
- 当前原型只在 loss 和 decode 阶段应用 sparse mask，没有减少 `GlobalPointer` 层内部的 dense token-pair score 矩阵计算。
- 因此，这版实现不能支撑“减少计算量/降低显存”的论文主张，只能作为早期失败原型记录。

### 后续方向

- 如果继续做 Sparse GlobalPointer，需要实现真正的候选级稀疏计算：
  - 先预测边界或候选 span。
  - 只 gather top-k span 的起止向量。
  - 只对候选 span 计算类别分数。
  - 避免构造完整 `[batch, entity_type, seq_len, seq_len]` 打分矩阵。
- 否则，当前 sparse mask 版本不建议作为核心创新继续使用。

## 2026-06-05 14:39:39 +08:00

### 本次新增改动

- 新增 `GP_EXPERIMENT_MODE=3`，用于运行 `Original GlobalPointer + Span Graph`。
- mode 3 当前是第一版 span graph 解码原型：
  - 训练阶段仍使用原版 GlobalPointer loss。
  - 验证和预测阶段从 GlobalPointer dense scores 中选取 top-k 候选 span。
  - 对候选 span 构建关系图。
  - 根据结构邻居分数计算 residual score。
  - 使用 `final_score = gp_score + lambda * graph_residual` 进行实体筛选。

### Span Graph 当前建模关系

- 包含关系：一个 span 包含另一个 span。
- 被包含关系：一个 span 被另一个 span 包含。
- 重叠关系：两个 span 有交集但不是包含关系。
- 共享起点：两个 span 的 start 相同。
- 共享终点：两个 span 的 end 相同。

### 新增环境变量

- `GP_GRAPH_TOPK`：每条样本进入 span graph 的候选 span 数量，默认 `256`。
- `GP_GRAPH_LAMBDA`：图 residual 分数权重，默认 `0.2`。
- `GP_GRAPH_ISOLATED_PENALTY`：孤立候选 span 的惩罚强度，默认 `0.5`。

### 使用方式

```python
import os
os.environ['GP_EXPERIMENT_MODE'] = '3'
```

然后运行：

```bash
python /kaggle/input/datasets/megafangyangpiy/gpgpgpgpgp/CMeEE_GlobalPointer.py
```

### 设计说明

- mode 3 第一版先不引入可训练 GNN，以降低不稳定性。
- 当前版本会从所有合法 span 中选取 top-k 候选，并允许图 residual 上调或下调候选分数。
- 当前目标是验证“结构关系后处理”是否能改变验证集 F1。
- 如果 mode 3 相比原版 GlobalPointer 没有提升，再考虑是否需要做可训练 span graph，而不是继续在后处理上调参。

## 2026-06-05 15:01:04 +08:00

### Mode 3 单 epoch 初步结果

- 实验设置：
  - `GP_EXPERIMENT_MODE=3`
  - `GP_GRAPH_TOPK=256`
  - `GP_GRAPH_LAMBDA=0.2`
  - `GP_GRAPH_ISOLATED_PENALTY=0.5`
- 第 1 个 epoch 训练指标：
  - `loss = 0.8988`
  - `global_pointer_f1_score = 0.4578`
- 第 1 个 epoch 验证指标：
  - `valid f1 = 0.61128`
  - `precision = 0.70882`
  - `recall = 0.53734`

### 初步观察

- mode 3 没有出现运行错误，span graph 解码流程可以跑通。
- 相比此前原版 10 epoch 日志中的第 1 个 epoch 结果 `valid f1 = 0.60459`，mode 3 第 1 个 epoch 略高。
- 当前提升主要来自 precision，recall 仍偏低。
- 单 epoch 结果不足以证明 span graph 有效，建议至少继续跑完整 10 epoch，看 best valid f1 是否超过原版 `0.65665`。
