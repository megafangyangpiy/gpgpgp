# Implementation Log

## 2026-06-05 16:12:07 +08:00

### 修改范围

- `CMeEE_GlobalPointer.py`
  - 增加 `GP_ABLATION` 消融模式与独立环境变量开关，支持 `baseline`、`no_relation`、`no_consistency`、`no_structure_decoding`、`full`。
  - 增加 span 关系工具，显式区分 `contain`、`inside`、`crossing`、`disjoint`、`exact`，用于嵌套实体结构建模。
  - 在数据生成阶段标记参与嵌套关系的正例 span；原始 GlobalPointer 损失仍使用二值标签，保证 baseline 逻辑可对照。
  - 增加嵌套正例一致性损失，可通过 `GP_USE_CONSISTENCY_LOSS=0` 或 `GP_ABLATION=no_consistency` 关闭。
  - 增加结构感知解码：候选 span 会根据内外层支持、共享边界支持、交叉 span 惩罚重新打分；可通过 `GP_ABLATION=no_structure_decoding` 回退到原阈值解码。
  - 扩展验证指标，训练时输出 overall F1、nested-context F1、nested F1、inner F1、outer F1、flat F1、crossing error rate、boundary error rate。
  - 将测试集预测改为批量预测，减少逐条预测带来的时间开销。
- `GlobalPointer_research_innovation_plan.md`
  - 增加“当前代码实现映射”小节，明确第一版实现采用轻量结构感知路线：span-pair 关系先落在解码重打分阶段，不引入独立 relation classifier。

### 设计逻辑

- 原代码只把每个实体 span 作为独立正例训练和解码，没有显式利用嵌套实体中常见的“内层实体被外层实体包含”“共享起止边界”“交叉 span 多为错误”等结构信息。
- 本次修改不改变 GlobalPointer 主干结构，而是在标注、损失、解码、评估四个位置加入结构信息，方便后续做消融实验判断 F1 提升来自哪个模块。
- 默认 `full` 模式用于完整创新点实验；`baseline` 模式尽量恢复原始训练和解码行为，用作对照。

### 后续验证

- 需要运行语法检查。
- 需要在数据和预训练模型可用环境下分别跑 `baseline` 与 `full`，比较 overall F1 与 nested F1。

## 2026-06-05 18:23:10 +08:00

### 修改范围

- `CMeEE_GlobalPointer.py`
  - 修复 `full` 模式下训练 `loss: nan` 的数值稳定性问题：一致性损失现在会先裁剪 GlobalPointer logits，再计算 `softplus`，避免 masked span 位置触发 `0 * inf = nan`。
  - 将 `PRUNE_CROSSING` 默认值绑定到 `USE_SPAN_PAIR_RELATION`，使 `GP_ABLATION=no_relation` 默认不再使用 crossing pruning，保证消融更干净。
  - 收紧 full 默认结构解码超参：降低候选 margin、inner/outer support、shared-boundary support、一致性损失权重，减少结构模块对 baseline 预测分布的过度扰动。
  - 扩展启动日志，打印 pruning、结构权重、一致性损失权重和 clip 值，方便复现实验。

### 实验判断

- `baseline.txt` 中 baseline 训练 loss 正常，最佳 overall F1 为 `0.65781`。
- `full.txt` 中 full 训练从第 1 个 epoch 起出现 `loss: nan`，该结果不能作为有效论文实验结果。
- 下一轮优先重新运行 `full`，确认 loss 不再为 `nan`；baseline 可暂时沿用当前正常结果，最终论文对比时再统一复跑。

## 2026-06-05 18:29:46 +08:00

### 修改范围

- `CMeEE_GlobalPointer.py`
  - 增加运行日志文件保存：默认在 `GP_OUTPUT_DIR` 下生成 `{ablation}_run_{timestamp}.log`，同步保存终端输出。
  - 增加结构化指标日志：默认在 `GP_OUTPUT_DIR` 下生成 `{ablation}_metrics_{timestamp}.jsonl`，每个 epoch 追加一行训练日志、验证指标和当前 best F1。
  - 增加 `GP_RUN_LOG_PATH`、`GP_METRICS_LOG_PATH`、`GP_DISABLE_FILE_LOG` 环境变量，允许自定义日志路径或关闭文件日志。

### 使用方式

- 默认运行不需要额外参数，日志会自动写入 `GP_OUTPUT_DIR`。
- 如需固定文件名，可设置 `GP_RUN_LOG_PATH` 和 `GP_METRICS_LOG_PATH`。
- 如只想终端输出，可设置 `GP_DISABLE_FILE_LOG=1`。

## 2026-06-05 22:38:05 +08:00

### 修改范围

- `CMeEE_GlobalPointer.py`
  - 修复运行日志保存时报错 `open.__init__() got an unexpected keyword argument 'buffering'`。
  - 原因是脚本中 `from bert4keras.snippets import open` 覆盖了 Python 内置 `open`；日志文件写入现在显式使用 `builtins.open`。
  - 同步将 epoch 指标 `.jsonl` 写入也改为 `builtins.open`，避免同类问题。

### 验证重点

- 下一次运行时应能正常打印 `run_log_path` 与 `metrics_log_path`。
- 如果仍在 Kaggle notebook 中运行，日志文件会写入当前 `GP_OUTPUT_DIR`。

## 2026-06-06 09:28:29 +08:00

### 修改范围

- `CMeEE_GlobalPointer.py`
  - 删除此前未带来 overall F1 实质提升的结构感知方案代码，包括 span-pair 关系、嵌套正例一致性损失、结构感知解码、嵌套分组指标和相关消融开关。
  - 恢复原始 GlobalPointer baseline 训练标签、交叉熵损失、F1 指标和 `threshold=0` 解码逻辑。
  - 保留运行日志保存和 epoch 指标 `.jsonl` 记录功能，作为后续实验记录工具。
  - 修复清理过程中由乱码注释导致的 tokenizer 初始化粘连问题，确保 baseline 可运行。

### 实验判断

- 当前主代码不再默认包含结构感知实验方案。
- 后续如继续冲 overall F1，应在 baseline 基础上重新选择更有效的增强策略，而不是沿用本次已删除的结构方案。
