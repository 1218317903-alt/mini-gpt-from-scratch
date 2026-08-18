# Tiny Shakespeare 混合精度对照实验

运行日期：2026-08-18  
统一配置：[`configs/tiny_shakespeare_baseline.yaml`](../configs/tiny_shakespeare_baseline.yaml)  
FP32 基线报告：[`tiny_shakespeare_baseline.md`](tiny_shakespeare_baseline.md)

## 实验问题

在 RTX 4060 Laptop GPU 上，对当前 81.8 万参数的字符级 MiniGPT 使用 FP16 或 BF16，
能否在基本不损失验证质量的情况下提升训练吞吐并减少显存？

本实验一次只改变 `--precision`。三组均使用同一份 Tiny Shakespeare 数据、模型结构、
batch size 32、block size 64、固定学习率、seed 42、2,000 个 Optimizer Step 和
4,096,000 个训练 Token。未启用梯度累积、调度器、Dropout 或 `torch.compile`。

FP32 运行对应提交 `d6e5535`；FP16 与 BF16 对应提交 `0e528cb`。两者之间只有文档
变更，训练源码、配置、依赖锁和数据没有变化，且三次训练启动时 Git 均为 clean。

## 精度实现

| 模式 | 前向计算 | 参数与 Optimizer 状态 | Loss Scaling |
|---|---|---|---|
| FP32 | FP32 | FP32 | 无 |
| FP16 | CUDA autocast FP16 | FP32 | 动态 GradScaler |
| BF16 | CUDA autocast BF16 | FP32 | 无 |

BF16 的指数范围比 FP16 更大，本实验不需要 GradScaler。Checkpoint 保存的是 FP32
主参数。独立验证、测试和生成统一使用 FP32 执行路径，目的是比较训练后权重，而不是
混入不同推理精度的影响。

## 结果

| 训练精度 | 最终训练 Loss | 最佳验证 Loss | 测试 Loss | 训练时间 | Token/s | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 1.5803 | 1.6666 | 1.8800 | 28.84 s | 142,041 | 105.59 MiB |
| FP16 | 1.5802 | 1.6666 | 1.8800 | 30.18 s | 135,707 | 87.33 MiB |
| BF16 | 1.5797 | 1.6660 | 1.8813 | 29.39 s | 139,362 | 87.33 MiB |

相对 FP32：

| 训练精度 | 吞吐变化 | 峰值显存变化 | 绝对显存减少 |
|---|---:|---:|---:|
| FP16 | -4.46% | -17.29% | 18.26 MiB |
| BF16 | -1.89% | -17.29% | 18.26 MiB |

训练时间和吞吐来自训练器累计的 step 计时，不包含过拟合检查、验证和 Checkpoint
I/O。峰值指标是 PyTorch `max_memory_allocated`，不是操作系统看到的进程总显存。

## 固定 Prompt

三组最佳 Checkpoint 均使用 FP32 推理，Prompt 为 `ROMEO:`，参数为
`temperature=0.8`、`top_k=20`、`seed=42`、生成 120 个字符。三组输出在这一个固定
样例上完全一致：

```text
ROMEO:
A thine my lords, for his leave.

DUKE ONGED OF YORK:
What, he ware unhe brince ountand sears, you beher'd;
So, the kee
```

这只能说明该 Prompt、seed 和采样设置下没有观察到差异，不能推出模型分布完全相同。

## 结论

1. 三种精度在 2,000 步预算下的 Loss 基本等价，没有出现 NaN、Inf 或训练发散。
2. FP16 和 BF16 都把 allocated 峰值显存降低约 17.29%，但绝对只减少约 18 MiB；
   原因是当前模型很小，固定运行时开销占比较高。
3. 这次单次运行中，FP16 和 BF16 都没有获得吞吐提升，分别比 FP32 慢 4.46% 和
   1.89%。小矩阵、autocast 转换和 FP16 GradScaler 的额外开销可能抵消 Tensor Core
   收益。
4. 在当前模型上，FP32 是速度最好的默认值；需要节省训练显存时优先选择 BF16，
   因为它与 FP16 显存相当、单次吞吐更高，并且不需要 Loss Scaling。

第 3、4 点是当前机器上的单次运行观察，不是跨硬件结论。若要把吞吐差异作为正式
性能结论，应对每种精度进行预热后至少重复 5 次，并报告均值、标准差和 GPU 功耗状态；
若要比较模型质量，应补至少 3 个 seed。

## 复现命令

```powershell
uv sync --locked --extra dev

uv run python train.py `
  --config configs/tiny_shakespeare_baseline.yaml `
  --run-dir runs/tiny-shakespeare-baseline-fp16-seed42 `
  --checkpoint-dir runs/tiny-shakespeare-baseline-fp16-seed42/checkpoints `
  --precision fp16

uv run python train.py `
  --config configs/tiny_shakespeare_baseline.yaml `
  --run-dir runs/tiny-shakespeare-baseline-bf16-seed42 `
  --checkpoint-dir runs/tiny-shakespeare-baseline-bf16-seed42/checkpoints `
  --precision bf16
```

每个本地 `runs/<name>/` 目录都保留最终生效配置、环境快照、2,000 个 step 指标、
训练日志和最佳/最新 Checkpoint；`runs/` 被 Git 忽略，仓库只提交轻量结果报告。

## 后续实验

固定学习率、Warmup、Warmup + Cosine 对照已经完成，见
[`learning_rate_schedule_comparison.md`](learning_rate_schedule_comparison.md)。阶段七最终
推荐方案与验收结果见 [`stage7_completion.md`](stage7_completion.md)。
