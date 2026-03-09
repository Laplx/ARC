# SDPO on ARC：训练指导文档

## 📌 项目目标

使用 Self-Distillation Policy Optimization (SDPO) 方法，在 ARC 数字序列补全任务上，**从已有的 16 个采样答案中提取训练信号**，提升模型的单次采样准确率（pass@1）。

**当前基线**：

- pass@1：14.75%
- pass@16（去重后至少一个正确）：25%

**目标**：训练后 pass@1 或 pass@16 提升。

---

## 🧠 核心思想（不要误解）

### ✅ 要做什么

对于每个有正确答案的题目：
- 取**错误答案**（由题目变换后进行采样生成）
- 把**正确答案**加入上下文中。注意只训练候选集里有正确答案的题目，若无，则跳过。
- 让模型重新 forward 这个错误答案，计算**如果早知道正确答案，每个 token 应该怎么选**
- 用 KL 散度让学生模型的分布向这个“事后分布”靠拢

### ❌ 不要做什么

| ❌ 错误做法                 | ✅ 正确做法                                                   |
| -------------------------- | ------------------------------------------------------------ |
| 让教师生成正确答案         | 考察教师的输出是**错误答案**，不是正确答案                   |
| 把正确答案当作要预测的目标 | 正确答案只作为**上下文**，不作为 labels                      |
| 只训练有正确答案的题目     | ✅ 对，但要用这些题目的**所有错误答案**                       |
| 所有题目都训练             | ❌ 只有那些**至少有一个正确答案**的题目（即候选集命中）才能产生训练数据 |
| 用正确答案做 SFT           | 这是不同的方法，不是 SDPO                                    |

---

## 📦 数据准备

### 输入数据

你已有的数据：（举例）不一定要严格按照下面的格式传递数据。
```python
{
    'question': 'Ċ86Ċ64',  # 用户输入
    'samples_16': [  # 至多 16 个采样答案（含一个正确的）
        'Ċ868686Ċ646464',           # 错误
        'Ċ868686Ċ646464Ċ686868',     # 错误
        ...  # 共 16 个或更少
    ],
    'ground_truth': 'Ċ868686Ċ646464Ċ686868Ċ464646Ċ868686Ċ646464'  # 已知正确答案
}
```

### 筛选可训练题目

（伪代码示意，不必完全遵照）

```python
trainable_questions = []
for q in all_questions:
    correct_samples = [s for s in q.samples_16 if s == q.ground_truth]
    wrong_samples = [s for s in q.samples_16 if s != q.ground_truth]
    
    # 必须同时有 1 个正确 + 至少 1 个错误
    if correct_samples and wrong_samples:
        trainable_questions.append({
            'question': q.question,
            'ground_truth': q.ground_truth,
            'wrong_samples': wrong_samples,
            # 注意：正确样本本身不作为训练数据
        })
```

**预期**：经预先评测，约 25% 的题目满足条件。

---

## 🔄 训练数据构造

对于每个 `(question, wrong_answer, ground_truth)` 三元组：

### 1. 学生输入

```
模板：{question（题目的示例pair，问题pair的input} {wrong_answer}
```

**作用**：计算模型在**没有 hindsight** 时对 wrong_answer 的分布。

### 2. 教师输入

```
模板：{question} {ground_truth pair（问题pair的input和正确output）} {问题pair的input（again）}  {wrong_answer}
```

**注意**：

- 教师的目标序列和学生的**完全一样**（都是 wrong_answer）
- 教师只是**上下文多了 ground_truth**
- 问题必须重现，否则模型不知道当前在评估哪个问题

---

## 🧮 Loss 计算

### 核心公式

```python
# 对 wrong_answer 的每个 token 位置 t
loss_t = KL_divergence(
    student_logits[t],     # 学生分布（无 hindsight）
    teacher_logits[t]      # 教师分布（有 hindsight）
)

loss = mean(loss_t)  # 对序列取平均
```

### 实现细节

示例代码，不用完全遵照。

```python
def compute_sdpo_loss(student_logits, teacher_logits, labels):
    """
    Args:
        student_logits: [seq_len, vocab_size] 学生输出
        teacher_logits: [seq_len, vocab_size] 教师输出
        labels: [seq_len] wrong_answer 的 token ids
    Returns:
        loss: scalar
    """
    # 只计算 wrong_answer 部分的 loss（忽略 prompt 部分）
    loss_mask = (labels != -100)  # 假设 -100 是 ignore index（如果需要 Mask 的话）
    
    # 对每个 token 计算 KL
    student_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
    
    token_kl = F.kl_div(
        student_probs,
        teacher_probs,
        reduction='none'
    ).sum(-1)  # [seq_len]
    
    # 只在 wrong_answer 位置取平均
    loss = (token_kl * loss_mask).sum() / loss_mask.sum()
    return loss
```

### 重要：stop gradient

```python
teacher_logits.detach()  # 教师不更新梯度
```

如果不 detach，教师会向学生退化，失去 hindsight 的作用。

---

## ⚙️ 训练参数

| 参数                  | 推荐值         | 说明                                            |
| --------------------- | -------------- | ----------------------------------------------- |
| batch size            | 32             | 每步处理的题目数                                |
| rollouts per question | 8              | 论文值，可以后面由我手动改成 16 个采样          |
| learning rate         | 1e-5           | 常量，不衰减                                    |
| optimizer             | AdamW          |                                                 |
| weight decay          | 0.01           |                                                 |
| gradient clip         | 1.0            |                                                 |
| warmup steps          | 10             | 小数据集可以更少                                |
| distillation loss     | Jensen-Shannon | 比 forward KL 更稳定                            |
| top-K distillation    | 100            | 节省显存，只计算 top-100 tokens，后面可以再调整 |

---

## 📊 验证方式（可选择开不开启）

### 验证集
- 与训练集**不重叠**的 ARC 题目（评估集取少量）（评估集和训练集由我设置目录来确定）
- 每 100 训练步评估一次

### 评估指标
- **pass@1**：单次采样准确率（重点）
- **pass@16**：16 次采样去重后准确率

### 验证时采样参数
```
temperature: 0.7
top-p: 0.95
num_return_sequences: 16  # 计算 pass@16
# 最后按原本的逻辑打分选出一个回答计算 pass@1
```

---

## 🧪 快速验证（可选，请单独到另一个文件里）

正式训练前，可以做“单步测试”：

1. 取 100 条失败答案
2. 分别计算：
   - 学生 log-prob(wrong_answer | question)
   - 教师 log-prob(wrong_answer | question + ground_truth)
3. 如果教师的 log-prob **显著更高**，说明 hindsight 有效，SDPO 会工作。
