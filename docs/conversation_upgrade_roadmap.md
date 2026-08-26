# MiniGPT 对话产品总体升级路线图

## 1. 目标

把当前“可调用的模型与 Agent 工程演示”升级为一个可以稳定进行多轮对话、展示生成过程、
保存会话、调用受控工具并能持续评测的本地 AI 助手。

目标不是让从零训练的 Tiny Shakespeare 模型承担通用聊天，而是保留两条清晰主线：

1. `src/minigpt/` 继续承担 Transformer、Tokenizer、训练和推理机制的研究教学主线。
2. Hugging Face Instruct 模型承担真实对话主线，并复用现有 LoRA、RAG、工具、文档和领域
   Adapter 能力。

## 2. 当前基线（v0.6.0-local-chat）

项目已经具备：

- `/chat`：接收完整消息数组的 Hugging Face 或 MiniGPT 聊天接口。
- `/agent/chat`：最多 6 步的受限工具调用循环。
- `/agent/chat/stream`：输出 route、model、tool、plan、result 等 SSE 阶段事件。
- `/demo`：可以发起单轮 Agent 请求的最小 HTML 页面。
- `/chat-ui`：SQLite 持久化的多会话聊天页面。
- `/conversations`：会话 CRUD、同步生成与逐 Token SSE 接口。
- 自动上下文压缩：保留 system 与最近轮次，较早消息写入持久化摘要；摘要失败时使用
  确定性提取策略。
- Request ID、健康检查、就绪检查、并发上限、取消令牌和进程内指标。
- 文档工具、检索、OCR/ASR/TTS 级联和可选私有领域 Adapter 控制面。

当前仍需继续补齐：Markdown 安全渲染、编辑后重试、回复分支、可选模型摘要器、聊天质量
评测、TTFT/P95 指标、普通聊天与 Agent 的统一事件协议，以及真实 GPU 取消延迟基准。

## 3. 目标架构

```text
Web Chat UI
  -> Conversation API
      -> Session Store (SQLite first)
      -> Context Builder / Token Budget
      -> Model Router
          -> HF Instruct / LoRA backend
          -> from-scratch MiniGPT research backend
      -> Agent Orchestrator
          -> Retrieval / Documents / Multimodal / Domain Adapter
      -> Token Event Stream (SSE)
      -> Trace + Metrics + Evaluation
```

第一版采用单机、单用户、本地优先设计。先用 SQLite 和 SSE，避免在需求尚未稳定时引入
Redis、消息队列、WebSocket 和复杂前端框架。

## 4. 分阶段实施

### C0：整理 GitHub 与可运行入口

交付：

- 使用 Conventional Commits，并让一个提交只表达一个可回滚能力。
- README 首页提供“一条命令启动聊天”的路径、架构图、演示截图和能力边界。
- 为仓库补充 description、topics、Issue/PR 模板和首个语义化 Release。
- CI 分成 `lint/typecheck`、`unit`、`service`、`slow/integration`，减少反馈时间。

验收：新用户在已准备模型缓存的机器上，10 分钟内能启动服务并完成一次 `/chat` 请求。

### C1：真正的多轮会话（已完成基础闭环）

交付：

- 新增 `ConversationRepository` 抽象和 SQLite 实现。
- 提供创建会话、列出会话、读取消息、追加消息、重命名和删除会话的 API。
- 每条消息保存 role、content、created_at、model、token usage、finish reason 和 request ID。
- 服务端构建上下文，不再要求浏览器每次提交全部历史。
- 超出 Token 预算时采用“保留 system + 最近轮次 + 可验证摘要”的压缩策略。

验收：服务重启后会话仍存在；连续 20 轮对话不会超过配置的上下文窗口；不同会话不串线。

### C2：逐 Token 流式生成与可靠取消（已完成基础闭环）

交付：

- 为聊天后端增加统一的 `stream()` 协议，HF 路径使用 streamer/生成线程输出 Token 增量。
- SSE 事件固定为 `start`、`delta`、`tool_start`、`tool_result`、`done`、`error`。
- 客户端断开或用户点击停止时设置 stopping criteria，使模型在下一个解码步退出。
- 保存已生成的部分回复，并明确标记 `cancelled` 或 `error`。

验收：首 Token 延迟、总吞吐和取消延迟均有自动化测试；取消后 GPU 任务能在短时间内释放。

### C3：可用的聊天界面（已完成第一版）

交付：

- 左侧会话列表，中间消息流，底部输入框，右侧可折叠 Agent trace。
- 支持 Markdown、代码块复制、停止生成、重新生成、编辑后重试和清空上下文。
- 展示当前模型、量化方式、Token 数、耗时、finish reason 和引用来源。
- 文件上传与图片/音频入口按服务能力动态显示，不可用能力明确禁用。

验收：桌面和移动宽度都可用；刷新页面不会丢失当前会话；错误能在消息级恢复。

### C4：RAG、工具与记忆分层

交付：

- 把“对话历史”“用户长期偏好”“检索文档”拆成三类上下文，分别设预算和来源标签。
- 检索结果必须携带文档 ID、片段 ID 和可校验引用；无证据时允许模型明确说不知道。
- 工具调用使用结构化 schema、幂等键、超时、重试预算和副作用确认。
- 长期记忆默认关闭；只保存用户明确允许的内容，并支持查看、修改和删除。

验收：冻结评测覆盖直接回答、检索、工具选择、参数生成、多步任务、恢复和引用忠实度。

### C5：质量、性能与发布

交付：

- 建立离线对话评测集：中文问答、多轮一致性、拒答、RAG、工具、长上下文和安全边界。
- 记录 TTFT、tokens/s、P50/P95 延迟、峰值显存、取消成功率和工具成功率。
- 比较 0.5B、1.5B、3B 以及 LoRA 前后结果；每次只改变一个关键变量。
- 增加版本化配置、数据库迁移、Docker Compose、Release notes 和回滚说明。

验收：发布前必须通过质量回归、服务契约、数据库迁移、并发与 GPU 显存门禁。

## 5. 推荐的近期执行顺序

1. 完成 C0 的 GitHub 展示与 Release 整理。
2. 为 C1–C3 增加真实 Qwen 3B/CUDA 端到端和取消延迟基准。
3. 完成 Markdown 安全渲染、重新生成、编辑后重试与消息分支。
4. 接入可选的模型摘要器，并保留当前确定性摘要作为可靠回退。
5. 在稳定会话协议上接入 C4，再以 C5 的评测结果决定模型和 LoRA 方案。

## 6. 建议的首个产品里程碑

`v0.6.0-local-chat` 建议只承诺以下范围：

- 单机单用户；
- Qwen2.5-3B-Instruct 4-bit 或本地等价 Instruct 模型；
- SQLite 多会话；
- SSE 逐 Token 输出与停止按钮；
- Markdown 聊天页面；
- 可选只读文档检索；
- 完整单元测试、API 契约测试和 10 轮端到端对话测试。

暂不承诺多租户、云部署、高可用、自动长期记忆或任意副作用工具。这个边界能让项目先形成
一个真实可演示、可测试、可迭代的对话闭环，再继续扩展。
