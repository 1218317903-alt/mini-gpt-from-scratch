# 分布式训练课程：DDP、FSDP 与当前硬件边界

## 1. 本节要学会什么

完成本节后，应能够：

1. 区分梯度累积、DDP 和 FSDP。
2. 解释 rank、local rank、world size、process group 和 backend。
3. 计算分布式训练的有效全局 Batch。
4. 说明 DistributedSampler 和 DDP 梯度同步的职责。
5. 读懂项目的 CPU/Gloo 双进程 smoke，并理解它能证明和不能证明什么。
6. 描述多 GPU 条件下继续实现 NCCL DDP、FSDP 和分片 Checkpoint 的路线。

## 2. 前置知识

当前硬件是单张 RTX 4060 Laptop 8GB。因此不能进行真实双 GPU NCCL 吞吐测试，也不能
实测 FSDP 的多卡显存收益。本项目采用两层证据：

```text
可以实测：CPU/Gloo 双进程 DDP 的梯度与参数同步
仅教学：GPU/NCCL 扩展效率、FSDP 分片显存和多卡 Checkpoint
```

这种边界必须写进报告，不能把 CPU 双进程或单 GPU 多进程包装成双 GPU 性能结果。

## 3. 核心内容讲解

### 3.1 梯度累积不是分布式训练

梯度累积让一个进程依次处理多个 micro-batch，累计梯度后再 optimizer step。它用更多时间
换取较低的瞬时显存。

DDP 则启动多个进程，每个进程通常绑定一张 GPU，各自处理不同数据并在 backward 时同步
梯度。它引入了真实进程间通信。

有效全局 Batch：

```text
global_batch = micro_batch × accumulation_steps × world_size
```

例如每卡 micro-batch 为 2，累积 8 次，共 4 个 rank，则全局 Batch 为 `2×8×4=64`。
对比单卡和多卡实验时必须保持全局 Batch、学习率策略和数据顺序尽量一致，否则 loss 差异不
能简单归因于分布式实现。

### 3.2 torchrun 环境变量

`torchrun` 为每个进程设置：

- `RANK`：全局进程编号。
- `LOCAL_RANK`：当前机器上的进程编号，常用于绑定 GPU。
- `WORLD_SIZE`：总进程数。

项目中的 `DistributedEnvironment.from_environment()` 读取并校验这三个值。rank 必须位于
`[0, world_size)`。

### 3.3 DDP 做了什么

DDP 中每个进程持有完整模型副本：

```text
rank 0: model copy + data shard 0
rank 1: model copy + data shard 1
...
```

各 rank 前向和本地 loss 可以不同；backward 期间 DDP 对梯度执行集体通信，使每个模型副本
看到相同的同步梯度。随后各 rank 使用同样的优化器状态和同步梯度更新，因此参数保持一致。

`DistributedSampler` 负责数据分片，不负责模型分片。每轮训练需要调用 `sampler.set_epoch`，
使所有 rank 在相同 epoch 规则下进行确定性 shuffle。

### 3.4 CPU/Gloo smoke 分析

运行：

```powershell
uv run python scripts/run_ddp_cpu_smoke.py
```

脚本启动两个 CPU 进程。两个 rank 使用不同输入，因此本地 loss 不同；经过 DDP backward
后，脚本 all-gather 梯度和更新参数并验证：

```text
maximum_gradient_difference_across_ranks = 0
maximum_parameter_difference_across_ranks = 0
```

输入是不同的数据，输出是同步梯度与一致参数。这说明 DDP 的核心语义工作正常。

它不能证明：

- 双 GPU 加速；
- NCCL 通信性能；
- GPU 利用率；
- FSDP 显存节省；
- 多机 RDMA/InfiniBand 行为。

### 3.5 FSDP 与 DDP

DDP 在每个 rank 保留完整参数、梯度和优化器状态。模型单卡放不下时，增加 DDP 卡数不会
自动解决单卡显存问题。

FSDP 的核心是分片训练状态：

```text
参数分片
梯度分片
优化器状态分片
```

计算某一层前，FSDP 按需聚合参数；使用后再释放或重新分片。这降低单卡显存，但增加通信、
wrap 策略和 Checkpoint 复杂度。FSDP 不保证必然比 DDP 快：模型能放进单卡时，DDP 可能有
更简单的通信路径和更高吞吐。

## 4. 有多卡时的下一步代码路线

### DDP

```text
torchrun --standalone --nproc_per_node=2 train_distributed.py
  -> init_process_group("nccl")
  -> torch.cuda.set_device(local_rank)
  -> DistributedSampler
  -> DDP(model, device_ids=[local_rank])
  -> rank 0 日志和 Checkpoint
  -> all_reduce 验证 loss
  -> destroy_process_group
```

验收必须包含：

- 各 rank 数据不重叠；
- 同步后的参数一致；
- 单卡与双卡相同全局 Batch 下 loss 接近；
- 记录 tokens/s、峰值显存和 scaling efficiency；
- 不能假设双卡就是 2 倍速度。

### FSDP

在 DDP 正确性完成后再增加：

- Transformer block auto-wrap；
- mixed precision policy；
- sharded state dict；
- full state dict 导出；
- resume；
- DDP/FSDP 显存与吞吐对照。

## 5. 本节总结

1. 梯度累积是单进程的时间换显存，不是分布式训练。
2. DDP 是数据并行，每个 rank 持有完整模型并同步梯度。
3. DistributedSampler 负责数据分片，DDP 负责梯度同步。
4. 全局 Batch 等于 micro-batch、累积次数和 world size 的乘积。
5. CPU/Gloo smoke 能验证进程和同步语义，不能证明 GPU 加速。
6. FSDP 分片参数、梯度和优化器状态，主要解决单卡显存问题。
7. 当前单 GPU 项目应诚实标注 FSDP/NCCL 未实测。

## 6. 自测题

1. 两个 DDP rank 的本地 loss 为什么可以不同，而 optimizer step 后参数仍然一致？
2. 如果双卡吞吐只有单卡的 1.4 倍，你会从模型规模、通信和数据加载哪些方面排查？
3. 模型单卡完全放不下时，为什么普通 DDP 不能解决问题，而 FSDP 可能可以？
