# Tiny Shakespeare 学习率调度对照实验

运行日期：2026-08-18  
统一配置：[`configs/tiny_shakespeare_baseline.yaml`](../configs/tiny_shakespeare_baseline.yaml)  
训练代码提交：`c4d8fb7`

## 实验问题

在 2,000 个 Optimizer Step、4,096,000 个训练 Token 的短预算下，线性 Warmup 和
Cosine Decay 是否优于固定学习率？

三组实验固定数据、模型、FP32、batch size 32、block size 64、seed 42 和峰值学习率
`3e-4`，只递进改变调度策略：

| 方案 | 前 200 步 | 后 1,800 步 |
|---|---|---|
| 固定 LR | `3e-4` | `3e-4` |
| Warmup | 从 `3e-5` 线性升至 `3e-4` | 保持 `3e-4` |
| Warmup + Cosine | 从 `3e-5` 线性升至 `3e-4` | Cosine 衰减至 `3e-5` |

## 结果

| 方案 | 最终训练 Loss | 最佳验证 Loss | 最佳 Step | 测试 Loss | 最终 LR |
|---|---:|---:|---:|---:|---:|
| 固定 LR | 1.5803 | **1.6666** | 2,000 | **1.8800** | `3e-4` |
| Warmup | 1.5995 | 1.6804 | 2,000 | 1.8878 | `3e-4` |
| Warmup + Cosine | 1.7161 | 1.7705 | 2,000 | 1.9438 | `3e-5` |

相对固定 LR，Warmup 的最佳验证 Loss 高 0.83%，Warmup + Cosine 高 6.24%。三组
最佳结果都出现在最后一步，没有观察到持续过拟合。

三次训练的累计 step 计时分别为 28.84、26.72 和 25.33 秒。调度器不会实质改变
模型计算量，这种单次运行速度差异更可能来自 GPU 动态频率和系统负载，因此不把它
解释为调度策略的吞吐收益。

## 固定 Prompt 样例

以下均使用最佳 Checkpoint、FP32 推理、`ROMEO:`、`temperature=0.8`、`top_k=20`、
`seed=42` 和 120 个新字符。

固定 LR：

```text
ROMEO:
A thine my lords, for his leave.

DUKE ONGED OF YORK:
What, he ware unhe brince ountand sears, you beher'd;
So, the kee
```

Warmup：

```text
ROMEO:
A shallem, your my father's corsempent'd and my this heart,
That king he bold with the of most of your mean
Sould faith
```

Warmup + Cosine：

```text
ROMEO:
A this emban to my father.

AUCINIO:
That will the bear for hearth.

DUCBOLOLHAM:
I fall as so you be to dady his faith
```

字符级生成的主观质量不足以覆盖 Loss 指标，样例只用于展示可复现输出，不据此单独
判定方案优劣。

## 结论与推荐方案

当前 2,000 步预算下推荐 **FP32 + 固定 `3e-4`**。Warmup 消耗了 10% 的短预算，
Cosine 又过早降低后期学习率，导致训练不足。调度器本身没有错；当模型更大、峰值
学习率更高、训练更长时，Warmup 可能改善早期稳定性，Cosine 也可能在充分训练后
改善收敛。

这是单 seed 结论。当前差异足以决定教学项目的默认配置，但若用于研究结论，应补
至少 3 个 seed 并报告均值和标准差。

## 复现命令

```powershell
# Warmup
uv run python train.py `
  --config configs/tiny_shakespeare_baseline.yaml `
  --run-dir runs/tiny-shakespeare-warmup-fp32-seed42 `
  --checkpoint-dir runs/tiny-shakespeare-warmup-fp32-seed42/checkpoints `
  --precision fp32 `
  --warmup-steps 200 `
  --warmup-start-factor 0.1

# Warmup + Cosine
uv run python train.py `
  --config configs/tiny_shakespeare_baseline.yaml `
  --run-dir runs/tiny-shakespeare-warmup-cosine-fp32-seed42 `
  --checkpoint-dir runs/tiny-shakespeare-warmup-cosine-fp32-seed42/checkpoints `
  --precision fp32 `
  --warmup-steps 200 `
  --warmup-start-factor 0.1 `
  --cosine-decay `
  --min-learning-rate 0.00003
```
