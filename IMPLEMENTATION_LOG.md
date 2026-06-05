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
