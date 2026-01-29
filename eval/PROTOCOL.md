## 暂定目录结构

  | 路径 | 作用 |
  |---|---|
  | evaluate.py | 唯一入口；解析参数、组装组件、启动评估 |
  | eval/ | 评估框架包 |
  | eval/__init__.py | 包导出与版本信息 |
  | eval/runner.py | 评估流程编排与调度 |
  | eval/metrics.py | 指标计算与聚合 |
  | eval/postprocess.py | 后处理与候选选择 |
  | eval/failure.py | 失败样本收集与归因 |
  | eval/visualize.py | 可视化与报告产出 |
  | eval/interfaces.py | Solver / Model / Dataset / Architect 接口定义 |
  | eval/registry.py | 可插拔组件注册与构建 |

  模块职责

  | 模块 | 职责 |
  |---|---|
  | evaluate.py | CLI 解析、配置加载、组件装配、运行入口 |
  | runner | 数据集迭代、调用 Solver、记录中间结果与日志、驱动指标与后处理 |
  | metrics | 任务级/集合级评估指标计算与汇总 |
  | postprocess | 候选解聚合、投票、多视角/多变换结果融合 |
  | failure | 失败样本分类、原因标注、保留复现所需最小信息 |
  | visualize | 生成失败分布、示例对比、指标曲线/表格 |

  最小接口（函数签名）

  | 模块 | 最小接口 |
  |---|---|
  | evaluate.py | def main(argv: list[str] | None = None) -> int: |
  | runner | def run(dataset, solver, metrics, postprocess, failure, visualize, *, config) -> dict: |
  | metrics | def compute(task, prediction, truth, *, context=None) -> dict:<br>def aggregate(records: list[dict], *,
  config=None) -> dict: |
  | postprocess | def collect(task, raw_outputs: list, *, context=None) -> list:<br>def select(task, candidates: list,
  *, context=None) -> object: |
  | failure | def record(task, prediction, truth, *, context=None) -> dict:<br>def summarize(failures: list[dict], *,
  config=None) -> dict: |
  | visualize | def build(reports: dict, *, output_dir: str) -> list[str]: |

  解耦接口（位于 eval/interfaces.py）

  | 角色 | 最小接口 |
  |---|---|
  | Dataset | def __iter__(self) -> "Iterator[Task]":<br>def info(self) -> dict: |
  | Solver | def solve(self, task, *, context=None) -> object: |
  | Model | def predict(self, inputs, *, context=None) -> object: |
  | Architect（可插拔） | def transform(self, task) -> list:<br>def score(self, task, candidate, *, context=None) ->
  float:<br>def select(self, task, candidates: list, *, context=None) -> object: |

## 注意

- 接口改为 def solve(self, task, *, context=None) -> Prediction

  candidates 中的每个元素必须至少包含：一个可用于最终比较的 grid 表示；可选的 program / prompt / trace；即 grid 是 evaluation 层的唯一硬语义。

- runner 不做任何策略判断，只做流程 orchestration

- context 仅用于：随机种子；运行配置（top-k、beam size）；日志 / trace 句柄；不允许在 context 中传递 task label 或 ground truth。

- metrics 中的 grid 对比函数：必须是 pure function；不依赖 solver / model / Architect；不包含任何 task-specific 规则。
  