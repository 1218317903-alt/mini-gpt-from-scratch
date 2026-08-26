# P8–P10 第一人称研究与验收报告

- 作者：李林知
- 日期：2026-08-25
- 范围：从零训练线现代化、图像/语音多模态、可选私有领域 Adapter
- 总结：P8/P9 工程骨架和受控实验完成；P10 治理与激活控制面完成，领域训练未执行

## 1. 我怎样控制本轮范围

P0–P7 已经证明纯文本主线能够训练、微调、检索、调用工具、评测、服务和生成真实文件。
所以 P8–P10 不再继续堆接口，而是回答三个更困难的问题：

1. 我能否在不破坏教学路径和旧 checkpoint 的前提下实现现代 Transformer 组件？
2. 图像和语音能否作为独立、可替换、可单独评测的阶段接入，而不是把所有错误归给 LLM？
3. 私有领域能力能否默认关闭、需要授权、隔离数据和 Adapter，并在训练前拒绝来源、成年人、
   同意或隐私不合格的数据？

我继续遵守 40GB 上限。P8 没有下载模型；P9 下载的 TrOCR small 与 Whisper tiny 加缓存共
383.99MiB，放在 D 盘；P10 没有领域语料或权重下载，也没有为了“展示成人能力”临时制造
训练数据。

## 2. P8：从零训练线现代化

### 2.1 兼容式架构设计

我没有另写一套无法和旧模型对照的“modern GPT”，而是给同一个 `MiniGPT` 增加五个默认
保持经典值的配置开关：

| 维度 | 经典默认 | 现代选项 |
|---|---|---|
| 位置编码 | learned absolute | RoPE |
| 归一化 | LayerNorm | RMSNorm |
| FFN | GELU | SwiGLU |
| Attention | 手写 causal attention | PyTorch SDPA |
| QKV | 三个独立 Linear | fused QKV Linear |

旧 YAML 不写新字段时仍走原路径；新配置 `configs/tiny_shakespeare_modern.yaml` 可以直接被
训练、评估、生成和服务加载。RoPE 在 KV Cache 增量解码时使用 past length 作为位置偏移；
SDPA 在 `Q<K` 的 cache 场景使用显式因果 mask，避免直接 `is_causal=True` 的对齐歧义。
当调用者要求 `output_attentions=True` 时，模型自动回到手写路径，因为 SDPA 不返回可解释
Attention 权重。

我另外实现了可保存、可恢复、确定性的教学版 BPE，以及带 EOS、尾部 `-100` loss mask 和
利用率统计的 sequence packing dataset。CrossEntropy 和梯度累积改为忽略 padding target，
避免把 padding 当成监督信号。

### 2.2 300 步固定预算结果

所有架构变体固定 seed 42、batch 16、block 64、4 层、4 heads、embedding 128、300 步、
307,200 个训练 Token，并使用同一个固定训练 Batch 和验证 Batch。这个设计适合测数学路径、
吞吐和过拟合行为，不等于 2,000 步正式随机 batch 训练。

| 变体 | 参数量 | tokens/s | 相对经典 | 峰值显存 | 最终验证 Loss |
|---|---:|---:|---:|---:|---:|
| classic | 818,432 | 93,928 | 基线 | 65.75MiB | 5.2934 |
| RoPE only | 810,240 | 72,017 | -23.33% | 65.69MiB | 4.9270 |
| RMSNorm only | 817,280 | 82,222 | -12.46% | 69.71MiB | 5.2731 |
| SwiGLU only | 1,078,016 | 95,500 | +1.67% | 85.72MiB | 5.5367 |
| SDPA only | 818,432 | 114,025 | +21.40% | 61.86MiB | 5.3227 |
| fused QKV only | 818,432 | 108,539 | +15.55% | 65.75MiB | 5.2786 |
| modern combined | 1,068,672 | 69,383 | -26.13% | 85.73MiB | 5.5207 |

经典模型和所有变体都把固定训练 Batch 的 Loss 压到约 0.025～0.033，但验证 Loss 反而从
约 4.2 上升到 4.93～5.54。这不是“现代架构失效”，而是固定 Batch 过拟合。它提醒我：
吞吐 benchmark 与泛化实验必须分开，不能拿训练 Loss 或单一 micro-benchmark 宣称精度提升。

本机短预算下，SDPA 是最清晰的性能收益：参数量不变，吞吐 +21.40%，峰值分配显存减少
约 3.89MiB。fused QKV 同样参数量不变，吞吐 +15.55%。组合现代架构反而慢 26.13%，因为
SwiGLU 增加约 30.6% 参数，加上当前小序列下 RoPE/Python 开销抵消了 kernel 收益。

### 2.3 BPE 结果

在训练文本前 100,000 个字符上，字符词表为 66；BPE 词表 256，编码后为 51,344 Token，
平均 1.9476 字符/Token，Token 数减少 48.656%，decode roundtrip 完全一致。这只证明压缩，
尚未证明相同字符/FLOPs 预算下的 BPB 更好；下一轮正式 P8 训练应使用随机 batch 和相同字符
预算比较验证 BPB。

证据：`docs/p8_architecture_benchmark.json`。

## 3. P9：图像与语音多模态

### 3.1 我为什么采用级联架构

在 8GB 显存和 40GB 总预算下，直接训练端到端 VLM 不符合资源边界和实验可信度。我把 P9
拆成 Image/OCR、ASR、Text Agent、TTS 四个可替换阶段，并给每一段独立计时。这样 OCR 错误、
ASR 错误、LLM 错误和 TTS 错误不会被混成“多模态回答失败”。

`MultimodalWorkspace` 限定 PNG/JPEG/WebP/WAV、纯文件名、目录范围、20MB 图片和 50MB 音频，
拒绝路径穿越和静默覆盖。服务增加 base64 上传、能力查询、图片分析、音频转写和语音合成
端点；Agent 增加 `analyze_image`、`transcribe_audio`、`synthesize_speech` 三个工具，其中 TTS
属于副作用工具，需要确认。

`MultimodalConversationPipeline` 可以组成图片 grounded question 和 ASR→LLM→TTS 语音闭环，
并分别返回 vision/OCR、ASR、LLM、TTS 耗时。后端使用 Protocol 注入，测试不下载权重；只有
显式设置 `MINIGPT_ENABLE_OCR/ASR/TTS=1` 才加载真实模型。

### 3.2 真实本地模型 smoke

我用 `microsoft/trocr-small-printed`、`openai/whisper-tiny` 和 Windows 离线语音合成执行了
真实 smoke，不是 mock：

| 环节 | 输入与结果 | 指标 |
|---|---|---:|
| OCR | `MINIGPT P9 TEST` → 完全一致 | CER 0.00% |
| TTS | `Mini GPT multimodal test` → 3.189 秒 WAV | 140,666 bytes |
| ASR | 合成 WAV → `MiniGPT Multimodo Test` | 字符错误率 9.52% |
| OCR 冷启动 | 包含模型加载 | 13,916.51ms |
| OCR 热路径 | 同进程第二次 | 75.56ms |
| ASR 冷启动 | 包含模型加载 | 6,528.43ms |
| ASR 热路径 | 同进程第二次 | 74.09ms |
| TTS | 本机系统语音 | 525.10ms |

首次 TrOCR 运行还暴露出 Transformers 5 tokenizer 转换缺少 `sentencepiece`/`protobuf` 的环境
问题。我把它们加入锁文件，并改为显式组合 DeiT PIL image processor 与 XLM-R tokenizer，
避免依赖有问题的自动 Processor 推断。这次失败被保留为环境可复现性的工程证据。

这些结果只覆盖清晰单行印刷体和系统合成短英文，不能外推到复杂截图、表格、中文扫描件、
方言或噪声语音。真实 LLM 的端到端 voice-to-voice 仍需单独负载测量；目前组件闭环和 Agent
组合路径已完成，但没有宣称联合多模态训练。

证据：`docs/p9_multimodal_smoke.json`。

## 4. P10：可选私有领域 Adapter

### 4.1 默认模型和领域模型隔离

我实现了版本化 `DomainAdapterManifest`、受限路径 Registry、SHA-256 Token 授权、常数时间
比较、公开安全 catalog、Adapter controller 和 Hugging Face PEFT 动态加载/切换接口。领域
根目录必须由 `MINIGPT_DOMAIN_ROOT` 显式指定；默认没有 Registry，公开模型不会自动加载任何
领域 Adapter。

服务提供 catalog、authorize 和请求级 Adapter 推理。Catalog 不返回本地路径或 Token hash；
调用方必须在单次 `/chat` 或 `/agent/chat` 中同时提供 `domain_id` 与
`domain_access_token`。服务在同一串行事务中完成授权、激活、推理和恢复，普通请求也不会与
领域请求重叠。旧的持久 activate/deactivate 路由返回 409，防止多个请求共享可变 Adapter
状态；生产环境仍应增加外部鉴权和按租户 worker 隔离。

### 4.2 数据进入训练前的硬门禁

每条领域 SFT 记录必须包含 record ID、prompt/response、来源、许可证、同意依据和个人数据
审查状态。含个人数据但没有完成审核会在 Schema 阶段失败；重复 ID 和静态禁止项会让整个
dataset report 失败。

成人领域只是一个默认关闭的模板，不包含语料或权重。若未来启用，manifest 必须同时满足：

- `access_mode=restricted`
- `minimum_age=18`
- 强制同意证明
- 每条记录确认所有人物为成年人
- 每条记录确认自愿情境
- 拒绝涉及未成年人、儿童性内容、强迫或非自愿性内容的记录
- 独立隐私、来源、许可证和人工复核

我没有接受“不要声张这一部分”的做法。敏感能力可以不作为默认产品卖点，但安全边界、数据
来源和访问控制不能隐藏，否则项目在公开研究和真实工程中都不可审计。

### 4.3 治理 smoke

合成标准领域记录通过；满足成年人/同意条件的合成成人治理记录通过；包含禁止标记的记录被
拒绝；错误 Token 被拒绝，正确 Token 可以选择标准领域；Catalog 不泄露路径和 Token hash；
默认公开模型未修改。

状态为 `governance_smoke_no_adapter_training`：P10 的隔离、治理和激活骨架完成，但没有真实
领域语料，也没有训练或评测领域 Adapter，因此不能宣称任何领域生成质量。

证据：`docs/p10_domain_governance_smoke.json`。

## 5. 当前结论

- P8：组件、训练/评估/服务兼容、测试和 300 步受控实验完成；2,000 步随机 batch 的现代
  配方正式泛化实验仍待执行。
- P9：受控上传、OCR/ASR/TTS 后端、Agent 工具、服务接口、分段延迟和真实模型 smoke 完成；
  复杂图像、真实噪声语音和真实 3B voice-to-voice 负载评测未完成。
- P10：Manifest、数据治理、授权、路径隔离、PEFT 激活控制面和 smoke 完成；真实领域 Adapter
  训练未执行，成人能力默认关闭且没有语料/权重。

所以本轮应表述为“完成 P8–P10 工程与受控验收，保留正式训练和困难数据集评测缺口”，而不
应该表述为“已经训练出端到端多模态或成人领域模型”。
