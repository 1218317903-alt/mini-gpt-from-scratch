# 后训练课程：从 SFT 到 DPO

## 1. 本节要学会什么

完成本节后，应能够：

1. 区分预训练、SFT 和偏好优化解决的问题。
2. 解释偏好样本中 prompt、chosen、rejected 的作用。
3. 从公式理解 DPO 为什么同时使用 policy model 和 reference model。
4. 读懂项目中的 response-only mask、序列对数概率和 DPO 训练循环。
5. 在 RTX 4060 Laptop 8GB 上运行小规模 0.5B 4-bit LoRA DPO smoke。
6. 明确两步 smoke、训练 loss 和真实质量提升之间的边界。

## 2. 前置知识

项目已有 SFT 管线：给定 system/user/assistant 消息，只对 assistant response 位置计算
交叉熵。SFT 教模型“对输入应输出什么”，但一条 prompt 通常只有一个示范回答，不能直接表达
“回答 A 比回答 B 更好多少”。

DPO 使用成对偏好：

```text
prompt   = 用户问题和必要上下文
chosen   = 偏好的回答
rejected = 相对较差的回答
```

例如，关于项目是否支持分布式训练，诚实说明“目前单卡，DDP/FSDP 未实测”应作为 chosen；
把 `accelerate` 依赖或 `device_map` 误称为完整分布式训练应作为 rejected。

## 3. 核心内容讲解

### 3.1 DPO 解决什么问题

SFT 最大化训练答案的似然，但真实偏好常常表现为比较：准确回答优于夸大回答，有引用的回答
优于编造回答，遵循用户格式的回答优于忽略约束的回答。DPO 直接利用这种比较信号。

直观上，DPO 希望 policy model 相对于 reference model：

- 更愿意生成 chosen；
- 更不愿意生成 rejected；
- 又不能无约束地远离 reference model。

reference model 通常是 SFT 后的冻结模型。项目脚本支持用同一个 SFT Adapter 初始化 policy
和 reference；policy Adapter 可训练，reference Adapter 冻结。如果不提供 SFT Adapter，脚本也
可以使用 LoRA policy 对比冻结基座，但这更适合教学 smoke，而不是正式对齐实验。

### 3.2 DPO 公式

项目实现的核心间隔为：

```text
r_chosen   = β [log πθ(chosen|x)   - log πref(chosen|x)]
r_rejected = β [log πθ(rejected|x) - log πref(rejected|x)]
margin     = r_chosen - r_rejected
L_DPO      = -log σ(margin)
```

符号含义：

- `x`：prompt。
- `πθ`：正在训练的 policy model。
- `πref`：冻结的 reference model。
- `β`：偏好间隔的缩放系数，不是学习率。
- `σ`：Sigmoid。
- `log π(y|x)`：只在 assistant response Token 上求和的序列对数概率。

简单代入：假设 `β=0.1`，policy 相对 reference 对 chosen 的改善为 `3`，对 rejected 的改善为
`1`，则 `margin=0.1×(3-1)=0.2`，loss 为 `-log σ(0.2)`。如果 policy 反而更加偏好
rejected，margin 变成负数，loss 会增大。

易错点：DPO loss 下降只表示模型更适合当前偏好对，不证明知识、推理、安全和通用能力全部
提高。必须配合冻结评测检查回退。

### 3.3 数据编码

实现位于 `hf_lora/src/hf_lora/preference.py`。

输入：一条 preference JSON 对象和 Tokenizer。

输出：

```text
chosen_input_ids
chosen_attention_mask
chosen_labels
rejected_input_ids
rejected_attention_mask
rejected_labels
```

chosen/rejected 共用同一 prompt。`labels` 中 prompt 位置全部为 `-100`，只有 response Token
参与序列对数概率和 DPO loss。这避免模型通过重新学习 prompt 来降低 loss。

数据校验会拒绝：

- prompt 不是合法消息序列；
- prompt 不以 user 结束；
- chosen/rejected 为空；
- chosen 和 rejected 完全相同；
- 样本超过 `max_length`。

### 3.4 训练循环

`train_dpo` 每一步执行：

```text
取 paired batch
  -> reference chosen/rejected 前向（no_grad）
  -> policy chosen/rejected 前向
  -> 计算四组 response log probability
  -> 计算 DPO loss
  -> 梯度累积
  -> unscale / clip grad norm
  -> optimizer step
  -> optimizer 确实更新后才 scheduler step
```

训练记录包括：

- loss；
- preference accuracy；
- reward margin；
- gradient norm；
- learning rate；
- optimizer 是否真正更新；
- step 时间；
- 峰值显存。

FP16 smoke 曾暴露非有限梯度，因此 4060 配方改用 BF16，并增加非有限梯度硬失败。这是一次
重要的训练工程经验：训练脚本“运行结束”不代表参数真的更新，必须检查 GradScaler、梯度范数
和 optimizer_updated。

## 4. 在当前硬件上运行

先检测：

```powershell
uv run python scripts/check_training_capabilities.py
```

当前 RTX 4060 Laptop 8GB 推荐：

```powershell
uv run python hf_lora/scripts/train_dpo.py `
  --config hf_lora/configs/qwen2_5_0_5b_dpo_smoke.yaml `
  --model-source <本地Qwen2.5-0.5B-Instruct快照>
```

正式一些的短实验：

```powershell
uv run python hf_lora/scripts/train_dpo.py `
  --config hf_lora/configs/qwen2_5_0_5b_dpo_4060.yaml `
  --model-source <本地快照> `
  --sft-adapter hf_lora/artifacts/<已有SFT实验>/adapter
```

不要直接上 3B DPO：DPO 同时需要 policy/reference 前向，8GB 的余量显著小于普通单模型
QLoRA。先记录 0.5B 的峰值显存，再决定是否尝试更长序列或更多 LoRA 模块。

## 5. 本节总结

1. SFT 学习示范答案，DPO 学习成对偏好。
2. DPO 比较 policy 相对 reference 对 chosen/rejected 的变化。
3. response-only mask 保证只评价回答 Token。
4. beta 控制偏好间隔缩放，不等于学习率。
5. preference accuracy 和 reward margin 是诊断指标，不是完整质量结论。
6. 当前 8GB 硬件适合 0.5B 4-bit LoRA DPO，不适合直接承诺 3B DPO。
7. 正式实验必须从 SFT Adapter 初始化并运行冻结回归评测。

## 6. 自测题

1. 为什么 chosen/rejected 完全相同时 DPO 没有有效偏好信号？
2. policy chosen log probability 上升，为什么仍然不能只看这一项判断 DPO 是否成功？
3. 如果 BF16 配方仍出现非有限梯度，你会按什么顺序排查学习率、数据、长度和量化？
