# Tiny Shakespeare FP32 正式基线

运行日期：2026-08-18  
实验配置：[`configs/tiny_shakespeare_baseline.yaml`](../configs/tiny_shakespeare_baseline.yaml)  
训练代码提交：`d6e5535b7b5e07fdae091d9fa5d8b69055d939fd`

## 目标

在真实 Tiny Shakespeare 语料上建立一个可复现的字符级 MiniGPT 基线，供后续
FP16/BF16、学习率调度、Dropout 和模型结构实验按单变量原则比较。本实验不是为了
追求生成质量上限，而是固定数据、训练预算、随机种子和评估口径。

## 环境与数据

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8 GB |
| Python | 3.11.9 |
| PyTorch | 2.13.0+cu126 |
| CUDA Runtime | 12.6 |
| 精度 | FP32 |
| Tokenizer | 字符级，66 个 Token（含 `<unk>`） |
| 训练 / 验证 / 测试字节数 | 923,811 / 115,568 / 116,015 |
| Git 状态 | `main`，训练启动时 clean |

原始语料 SHA-256：
`a29d649defa42cc30feade39d794c74a4f54c23e7dac87b44ed9b9e3f20da95b`。
数据划分由仓库脚本确定，训练文件 SHA-256 为
`94ff9d9d1bb8f395f5d8ce1dcd3e32a11de916acbc9f6da1160fbff16fc11985`。

## 固定配置与预算

| 项目 | 值 |
|---|---:|
| block size | 64 |
| batch size | 32 |
| embedding dim | 128 |
| heads / layers | 4 / 4 |
| FFN expansion | 4 |
| 参数量 | 818,432 |
| learning rate | 0.0003，固定 |
| weight decay | 0.01 |
| dropout | 0.0 |
| seed | 42 |
| optimizer steps | 2,000 |
| 每步 Token | 2,048 |
| 总训练 Token | 4,096,000 |

未启用梯度累积、Warmup、Cosine、混合精度或 `torch.compile`。正式训练前的固定
单样本过拟合检查从 Loss `4.2064` 降至 `0.6969`，训练链路通过。

## 结果

| 指标 | 结果 |
|---|---:|
| 首步训练 Loss | 4.2321 |
| 第 2,000 步训练 Loss | 1.5803 |
| 最佳验证 Loss | 1.6666（step 2,000） |
| 测试 Loss | 1.8800 |
| 验证 / 测试 Perplexity | 5.29 / 6.55 |
| 训练步纯计算时间 | 28.84 s |
| 累计吞吐 | 142,041 Token/s |
| 峰值 allocated 显存 | 105.59 MiB |

验证和测试 Loss 均使用最佳 Checkpoint，并各评估固定的 50 个 Batch。训练时间与
吞吐来自训练器累计的 step 计时，不包含过拟合检查、验证和 Checkpoint I/O，因此
不应把它解释为端到端墙钟时间。

验证 Loss 总体从 `3.9464` 持续下降到 `1.6666`；step 1,800 曾小幅回升至
`1.7052`，之后继续下降。当前 2,000 步尚未出现持续过拟合，后续可以在不改变
其他变量的前提下测试更长预算。

## 固定 Prompt 样例

平衡采样统一使用 `temperature=0.8`、`top_k=20`、`seed=42`、生成 120 个字符。

Prompt `ROMEO:`：

```text
ROMEO:
A thine my lords, for his leave.

DUKE ONGED OF YORK:
What, he ware unhe brince ountand sears, you beher'd;
So, the kee
```

Prompt `KING HENRY:`：

```text
KING HENRY:
A shall my lord, so much necause: prited and my thee,
Which he waked his abour to prantor me ot.

KING RICHASTII:
The s
```

Greedy 解码，Prompt `To be, or not to be:`：

```text
To be, or not to be:
The world the ware the world the see the seast the worth
That the worther the wore the wore the wore of the shall of th
```

模型已经学到角色标签、换行、标点和局部戏剧文本结构，但仍会生成拼写伪词；Greedy
结果还有明显重复。字符级、小模型和 409.6 万 Token 预算下的结果符合基线定位，
不能据此声称具备自然语言理解能力。

## 复现

```powershell
uv sync --locked --extra dev

uv run python train.py `
  --config configs/tiny_shakespeare_baseline.yaml `
  --run-dir runs/tiny-shakespeare-baseline-fp32-seed42 `
  --checkpoint-dir runs/tiny-shakespeare-baseline-fp32-seed42/checkpoints `
  --precision fp32

uv run python evaluate.py `
  --config configs/tiny_shakespeare_baseline.yaml `
  --checkpoint runs/tiny-shakespeare-baseline-fp32-seed42/checkpoints/best.pt `
  --split test
```

`runs/` 和 Checkpoint 被 Git 忽略，避免把约 10 MB 的权重和逐步日志提交到代码仓库；
配置、指标摘要、复现命令和代表性输出保留在本报告中。本机原始运行目录包含
`config.yaml`、`environment.json`、`metrics.jsonl`、`train.log`、`best.pt` 和
`latest.pt`。

## 后续实验

同预算的 FP16、BF16 对照已经完成，见
[`mixed_precision_comparison.md`](mixed_precision_comparison.md)。下一项按单变量原则
比较固定学习率、Warmup、Warmup + Cosine。
