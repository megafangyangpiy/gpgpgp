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
