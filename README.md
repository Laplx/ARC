# ARC Prize 2025

- 评估框架

  IO 格式：

  Raw：
  - Mistral 7B
  - Qwen3 4B Thinking
  - Llama 3.2B/8B

  ARChitect：

  NVARC：

- 数据增强

  基础数据：ARC-AGI2

- Idea 框架

### 评估结构（压缩版）

| 路径 | 作用 |
|---|---|
| evaluate.py | 评估入口：解析参数、组装组件、调用核心评估 |
| eval/core.py | 数据集 + codec + 评估流程（记录/汇总/报告） |
| eval/models.py | HF 推理封装 |
| eval/solvers.py | 轻量 Solver（RawSolver） |
| scripts/architect_dfs.py | 独立的 Architect DFS 版本（不进入主评估链路） |

### 模块职责

| 模块 | 职责 |
|---|---|
| evaluate.py | CLI 解析、组件装配、运行入口 |
| eval/core.py | 评估主循环、指标汇总、失败报告与输出 |
| eval/models.py | 模型加载与生成 |
| eval/solvers.py | 基础推理策略 |
