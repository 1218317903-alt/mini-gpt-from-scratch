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

## 阶段七：工程化与可复现实验

1. Python 3.11、uv、pyproject 与锁文件统一环境
2. GitHub 公有仓库与自动 CI
3. Ruff 格式化、Lint 与 mypy 静态类型门槛
4. pytest、分支覆盖率与端到端训练测试
5. 配置、环境、Git 状态、JSONL 指标与日志追踪
6. 原子 Checkpoint、实验隔离与恢复训练
7. Tiny Shakespeare FP32 正式基线
8. FP32、FP16、BF16 单变量对照
9. 固定 LR、Warmup、Warmup + Cosine 单变量对照
10. README、实验报告、复现说明与阶段验收

## 当前状态

阶段一到七已完成。阶段一到五保留从零实现路径；阶段六位于独立的 `hf_lora/`，
不会改写 MiniGPT；阶段七完成工程化、真实基线和单变量实验。阶段八完成服务化，
阶段九完成真实聊天后端；下一阶段进入工具调用与 RAG。

## 阶段八：模型服务化

1. FastAPI 服务目录与应用骨架
2. 推理请求与响应模型
3. Checkpoint、Tokenizer 与模型生命周期管理
4. 文本生成接口
5. 健康检查与就绪检查
6. 结构化日志、请求追踪与异常处理
7. Docker 镜像与容器启动
8. 基础负载测试、性能记录与阶段验收

当前完成实现一：提供可复用的 FastAPI 应用工厂和 ASGI 应用入口。
当前完成实现二：定义 `GenerateRequest` 与 `GenerateResponse`，统一生成参数校验和响应结构。
当前完成实现三：封装 `ModelService`，负责配置、Tokenizer、Checkpoint、模型加载和生命周期释放。
当前完成实现四：注册 `POST /generate`，将生成参数传递给模型服务并返回统一响应；模型未就绪时返回 503。
当前完成实现五：注册 `/health` 存活检查和 `/ready` 模型就绪检查。
当前完成实现六：增加请求 ID、JSON 结构化访问日志和统一错误响应。
当前完成实现七：提供基于锁定依赖的 Docker 镜像、健康检查和环境变量启动配置。
当前完成实现八：加入基础 HTTP 负载测试、性能 JSON 记录和阶段验收文档。
后续实现按上述顺序推进。

## 阶段九：真实聊天后端

1. 统一 `ChatModel`、`ChatResult` 和消息角色契约
2. Hugging Face Instruct 模型与 Chat Template 推理
3. 可选 PEFT LoRA Adapter 加载与独立恢复
4. Greedy、Temperature、Top-k、Top-p、seed 和 EOS 停止
5. Prompt/生成 Token 统计与上下文长度校验
6. FastAPI `POST /chat` 与阶段八 `/generate` 并存
7. 环境变量配置、离线模型缓存和真实接口验收

当前完成：阶段九七项实现均已落地。下一阶段进入工具注册、工具执行和 Agent 循环，
随后加入 RAG。

## 阶段十：工具调用与最小 Agent 循环

1. `ToolSpec`、`ToolDefinition`、`ToolCall` 与执行结果契约
2. 显式工具注册表和重复注册保护
3. Pydantic 参数校验、异常标准化和工具超时
4. 安全计算器与当前时间工具
5. JSON 工具调用解析与普通聊天模型适配
6. 最大轮数、错误恢复和执行轨迹
7. FastAPI `POST /agent/chat` 接口

当前完成：阶段十七项实现均已落地。

## 阶段十一：统一聊天后端与 Agent 评测集

1. 把 `ChatModel` 收到 `src/agent/llm/`，MiniGPT / HF / LoRA 共用同一接口
2. `/chat` 维护多轮消息、EOS 停止、Token 与耗时统计
3. 0.5B 用于 CI，3B + 4-bit 作为 8GB 本地主模型
4. 建设 100 条独立评测集和 Recall@5 / 工具指标
5. 用评测目标约束后续 RAG 和多步恢复，而不是先写更复杂循环

当前完成：统一后端、第一版评测集、真实 Agent runner，以及 0.5B LoRA / 3B 4-bit
对照基线已落地。下一阶段在独立开发集上改进工具路由与引用回答，并把 0.5B smoke
接入 CI、3B 全量保留为本地回归。
