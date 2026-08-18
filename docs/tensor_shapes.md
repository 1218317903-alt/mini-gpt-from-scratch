# MiniGPT 张量形状

统一记号：

```text
B：Batch Size
T：上下文序列长度
C：Embedding 维度
H：Attention Head 数量
D：每个 Head 的维度，C = H × D
V：词表大小
```

## 阶段一：数据

单个 Dataset 样本：

```text
Token IDs：      [T+1]
x = window[:-1]  [T]
y = window[1:]   [T]
```

DataLoader 后：

```text
x：[B,T]
y：[B,T]
```

错位关系：

```text
y[:, :-1] == x[:, 1:]
```

## 阶段二：模型

```text
Token IDs              [B,T]
Token Embedding        [B,T,C]
Position Embedding     [B,T,C]
Q/K/V Projection       [B,T,C]
Split Heads             [B,H,T,D]
K Transpose             [B,H,D,T]
Attention Scores        [B,H,T,T]
Causal Mask             [B,H,T,T]
Attention Weights       [B,H,T,T]
Weights × V             [B,H,T,D]
Merge Heads             [B,T,C]
FFN Hidden              [B,T,expansion_factor×C]
FFN Output              [B,T,C]
Language Model Head     [B,T,V]
```

请求 `output_attentions=True` 时：

```text
MiniGPT output          (logits, attentions)
logits                  [B,T,V]
len(attentions)         L
attentions[layer]       [B,H,T,T]
attentions[layer][b,h]  [T,T]
```

默认不请求权重时仍只返回 `[B,T,V]` logits。

核心矩阵乘法：

```text
[B,H,T,D] × [B,H,D,T] → [B,H,T,T]
[B,H,T,T] × [B,H,T,D] → [B,H,T,D]
[B,T,C]   × Linear(C,V) → [B,T,V]
```

## 阶段三：训练

```text
inputs       [B,T]
targets      [B,T]
model output [B,T,V]
flattened logits  [B×T,V]
flattened targets [B×T]
cross entropy     scalar
```

训练时一次前向传播会得到所有位置的 logits；Causal Mask 防止每个位置看到未来。

## 阶段四：单步推理

```text
input_ids              [B,T]
model(input_ids)       [B,T,V]
select_last_logits     [B,V]
Temperature            [B,V]
Top-k / Top-p mask     [B,V]
Softmax probabilities  [B,V]
Greedy / sample        [B]
```

最后一个位置的 logits 表示：

```text
P(next_token | current_context)
```

## 阶段四：多步生成

每一步只追加一个 Token：

```text
generated              [B,T+n]
context = last window  [B,min(T+n,block_size)]
logits                  [B,T_context,V]
last_logits             [B,V]
next_token              [B]
next_token.unsqueeze(1) [B,1]
new generated           [B,T+n+1]
```

完整序列可以超过 `block_size`，但传给模型的 `context` 不会超过 `block_size`。

## 训练与推理模式

```text
训练：model.train() + 梯度 + loss.backward()
推理：model.eval()  + torch.no_grad()
```

## 阶段五：梯度累积

```text
Micro Batch inputs      [B_micro,T]
单次参数更新的批量       K 个 Micro Batch
有效 Batch Size         B_micro × K
每次更新的 Token 数      B_micro × T × K
```

每个 Micro Batch 完成一次前向和反向；全部 `K` 个 Micro Batch
完成后，才执行一次梯度裁剪和一次优化器更新。
# KV Cache 张量形状

阶段五把完整序列注意力推广为矩形注意力。设当前输入长度为 `Q`，历史缓存长度为 `P`，则总 Key 长度 `K=P+Q`：

| 张量 | 形状 | 含义 |
|---|---|---|
| 当前 Query | `[B,H,Q,D]` | 只为本次输入计算 |
| 新 Key/Value | `[B,H,Q,D]` | 本次输入产生的 K/V |
| 历史 Key/Value | `[B,H,P,D]` | 当前层过去的缓存 |
| 拼接后 Key/Value | `[B,H,P+Q,D]` | 返回给下一步的 Cache |
| 注意力分数/权重 | `[B,H,Q,P+Q]` | 当前 Query 查询全部可见 K/V |

模型 Cache 是长度等于 `num_layers` 的元组，每一项为 `(key, value)`。单 Token Decode 时 `Q=1`，所以注意力权重为 `[B,H,1,P+1]`。

位置编码使用 `position_offset=P`。第一版缓存推理要求 `prompt_length + max_new_tokens <= block_size`，因为直接截断带有可学习绝对位置编码的旧 Cache 与重新计算滑动窗口并不等价。
