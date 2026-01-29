# Stage 1

评估框架：
    - 使用扩充后数据集
    - 确定 token 种类和文本格式
    - 留下 LLM 接口
    - 失败（结果）可视化
    - 后处理优化
      - 将输入经过多种变换后投票
      - DFS，即复现 Architect 思路

LoRA 微调框架：用于测试时微调（TTFT）

评估基座和经过 TTFT 的结果

Idea：
    - PEARL
    - 快速原型与超参数探索
