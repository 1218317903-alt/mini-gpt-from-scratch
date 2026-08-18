# 阶段六：Hugging Face SFT 与 LoRA

本目录与从零实现的 `src/minigpt/` 完全隔离，用于把阶段一到五的底层知识迁移到真实预训练模型。

## 对照关系

| MiniGPT | Hugging Face / PEFT |
|---|---|
| `CharacterTokenizer` | `AutoTokenizer` 与子词词表 |
| `x/y` 显式错位 | `input_ids/labels`，模型内部 shift |
| Causal Mask | Causal LM 内部 Mask |
| 无 Padding | `attention_mask` 与动态 Padding |
| 全部 Token Loss | Assistant response-only labels |
| 随机初始化权重 | 真实预训练权重 |
| 全参数训练 | 冻结基座，只训练 LoRA A/B |

## 目录

```text
hf_lora/
├── configs/                 实验配置
├── data/                    人工审核的教学 JSONL
├── scripts/                 Tokenizer 检查与训练/恢复入口
├── src/hf_lora/             数据、模型、训练和生成模块
├── tests/                   默认不联网的单元测试
└── artifacts/               Adapter、指标和生成对比（Git 忽略）
```

## 环境

本目录与根 MiniGPT 共用项目根目录的 Python 3.11 虚拟环境和依赖锁文件。
请在项目根目录运行：

```powershell
uv sync --extra dev --extra hf
```

本机验收模型为 `Qwen/Qwen2.5-0.5B-Instruct`，8GB RTX 4060 Laptop 使用普通 FP16 LoRA，不使用 QLoRA。

## 数据格式

每行一个 JSON 对象：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

编码规则：

```text
System/User：        attention_mask=1, labels=-100
Assistant Response： attention_mask=1, labels=Token ID
Padding：            attention_mask=0, labels=-100
```

本教学实现对超长样本报错，不静默截掉回答或结束标记。

## Tokenizer 与聊天模板

```powershell
python hf_lora/scripts/inspect_tokenizer.py
```

严格离线时通过 `--model-source` 传本地 Hugging Face 快照目录。

## 训练、保存和恢复

```powershell
python hf_lora/scripts/train_and_compare.py `
  --config hf_lora/configs/qwen2_5_0_5b_lora.yaml
```

输出：

```text
hf_lora/artifacts/qwen2_5_0_5b_lora/
├── adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── comparison.json
└── training_metrics.json
```

脚本会执行训练前验证与生成、LoRA 微调、Adapter 保存、删除训练模型、加载全新基座、恢复 Adapter、logits 一致性检查以及固定 Prompt 前后对比。

## LoRA 与训练配置

```text
r=8
alpha=16
dropout=0.05
target_modules=[q_proj, v_proj]
bias=none
```

Qwen2.5-0.5B 共引入 540,672 个可训练参数，约占包装后模型的 0.1093%。

- Micro Batch 为 1，梯度累积为 4。
- 梯度按有效 Assistant Token 数加权，避免不同回答长度导致隐性权重偏差。
- 学习率从 `2e-4` Warmup 后 Cosine Decay。
- 本机 FP16 GradScaler 初始值使用 256，默认高 scale 曾造成首步梯度溢出。
- 训练时关闭 KV Cache；生成时临时开启。

## 测试

项目根目录直接运行：

```powershell
uv run pytest -q
```

根 `pyproject.toml` 会安装 `minigpt` 与 `hf_lora` 两个源码包，无需手动设置
`PYTHONPATH`，也不再维护目录内的第二份依赖清单。

## 局限

- 教学数据规模很小，只能验证闭环，不能证明通用能力提升。
- SFT 会模仿数据中的事实、风格与错误，数据质量决定上限。
- 0.5B 模型的基础推理和知识能力有限。
- 本阶段未使用 QLoRA、DeepSpeed、FSDP、RAG 或 Agent。

## 本机验收结果

在 RTX 4060 Laptop 8GB、FP16、16 条训练数据和 4 条独立验证数据上运行 60 个 Optimizer Step：

```text
基线验证 Loss：3.17293
最终验证 Loss：2.70556
总参数：494,573,440
可训练参数：540,672（0.109321%）
目标模块：48
峰值已分配显存：约 1,251.6 MiB
Adapter 恢复最大 logits 误差：0.0
```

任务外算术回答在微调前后都保持正确；领域回答的措辞有所变化，但仍存在概念不准确。这说明小样本 Loss 下降不能替代人工生成评估，后续应优先增加高质量释义、反例和组合问题，而不是无限增加训练步数。
