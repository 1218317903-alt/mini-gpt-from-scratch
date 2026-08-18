# mini-gpt-from-scratch 学习计划

## 阶段一：数据流

1. 原始文本读取
2. train/validation/test 按原始顺序划分
3. 字符级 Tokenizer
4. Tokenizer JSON 保存与加载
5. `x/y` 错位 Dataset
6. DataLoader 形成 `[B,T]`
7. 数据流测试与 Batch 检查

## 阶段二：Decoder-only MiniGPT

1. Token Embedding
2. Position Embedding
3. 独立 Q/K/V Projection
4. Scaled Dot-Product Attention
5. Causal Mask
6. Multi-Head Attention
7. FFN、GELU、Residual、LayerNorm
8. Pre-Norm Transformer Block
9. MiniGPT 与 Language Model Head
10. 可选权重共享
11. 前向形状、反向传播和 Batch 独立性测试

## 阶段三：训练与 Checkpoint

1. Teacher Forcing
2. `[B,T,V]` logits 与交叉熵
3. AdamW 优化和梯度裁剪
4. 验证 Loss
5. 极小 Batch 过拟合检查
6. best/latest Checkpoint
7. Tokenizer 信息与随机状态保存
8. 训练恢复和独立评估

## 阶段四：自回归生成与基础采样

1. 训练并行预测与推理逐 Token 生成
2. 最后位置 logits、Softmax、Greedy
3. 随机采样与随机种子
4. Temperature
5. Top-k
6. Top-p
7. 自回归循环、`max_new_tokens`、`block_size` 截断和空 Prompt
8. Checkpoint、Tokenizer、CLI、解码和端到端闭环

## 阶段五：分析与训练优化

1. 参数量和模块占比统计
2. Attention 热力图
3. 梯度累积、Warmup、Cosine 与混合精度
4. KV Cache 与缓存一致性
5. 消融实验入口

## 阶段六：Hugging Face SFT 与 LoRA

1. Transformers、Datasets、Tokenizers 与自动工厂
2. 子词 Tokenizer 与聊天模板
3. `input_ids`、`attention_mask`、`labels` 与 Causal LM Loss
4. Response-only SFT 数据和动态 Padding
5. LoRA 数学原理与参数推导
6. PEFT 注入、目标模块与真实参数统计
7. FP16 LoRA 训练、梯度累积、调度与验证
8. Adapter 保存、全新恢复和固定提示词前后对比

## 当前状态

阶段一到五保留从零实现路径；阶段六位于独立的 `hf_lora/`，不会改写 MiniGPT。默认测试覆盖两个模块，真实模型实验使用小型 Qwen Causal LM。
