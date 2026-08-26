"""Export the reproducible P0-P10 career-project report as DOCX and PDF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.document_tools import DocumentWorkspace

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(path: Path) -> dict[str, Any]:
    return _load_json(path)["scoring"]["metrics"]


def main() -> None:
    """Create both final artifacts through the validated document contract."""
    frozen = _load_metrics(ROOT / "docs/p0_p5_final_frozen_3b_qlora_eval.json")
    p6 = _load_metrics(ROOT / "docs/p6_agent_v1_1_3b_qlora_dev_eval.json")
    p8 = _load_json(ROOT / "docs/p8_architecture_benchmark.json")
    p9 = _load_json(ROOT / "docs/p9_multimodal_smoke.json")
    p10 = _load_json(ROOT / "docs/p10_domain_governance_smoke.json")

    p8_results = {row["variant"]["name"]: row for row in p8["architecture_results"]}
    classic_throughput = p8_results["classic"]["tokens_per_second"]
    metrics = [
        {
            "label": "P0-P5 Frozen 任务成功率",
            "value": f"{frozen['task_success_rate']:.2%}",
        },
        {
            "label": "P0-P5 Frozen RAG Recall@5",
            "value": f"{frozen['rag_recall_at_5']:.2%}",
        },
        {"label": "P6 Dev 任务成功率", "value": f"{p6['task_success_rate']:.2%}"},
        {"label": "P6 Dev Multi-step", "value": "93.33% (14/15)"},
        {
            "label": "P8 SDPA 吞吐变化",
            "value": (
                f"{(p8_results['sdpa_only']['tokens_per_second'] / classic_throughput - 1) * 100:+.2f}%"
            ),
        },
        {
            "label": "P8 fused QKV 吞吐变化",
            "value": (
                f"{(p8_results['fused_qkv_only']['tokens_per_second'] / classic_throughput - 1) * 100:+.2f}%"
            ),
        },
        {
            "label": "P8 BPE Token 减少",
            "value": f"{p8['tokenizer_comparison']['token_reduction_rate']:.3%}",
        },
        {
            "label": "P9 OCR / ASR 字符错误率",
            "value": (
                f"{p9['ocr_character_error_rate']:.2%} / "
                f"{p9['asr_character_error_rate']:.2%}"
            ),
        },
        {
            "label": "P9 热路径 OCR / ASR",
            "value": (
                f"{p9['warm_latency_ms']['ocr']:.2f} / "
                f"{p9['warm_latency_ms']['asr']:.2f} ms"
            ),
        },
        {
            "label": "P9 模型缓存",
            "value": f"{p9['model_cache_mib']:.2f} MiB（D 盘）",
        },
        {
            "label": "P10 治理 smoke",
            "value": (
                "通过"
                if p10["catalog_hides_paths_and_tokens"] and p10["unauthorized_denied"]
                else "失败"
            ),
        },
        {"label": "全量工程测试", "value": "256 passed"},
        {"label": "代码覆盖率", "value": "71.62%（门槛 60%）"},
    ]
    findings = [
        "P0/P1：我先固定范围、资源上限、Dev/Frozen 数据治理和污染检查，避免边调边看测试集。",
        "P2：我把自由生成与确定性 heuristic router 做同集对照，让路由收益能被单独解释。",
        "P3/P4：我构建多轮 response-only SFT 数据，并完成 0.5B LoRA 与 3B NF4 QLoRA 的保存、重载和对照。",
        "P5：我实现 lexical、dense、hybrid 与 weighted RRF；冻结结果是任务成功率 74.00%、RAG Recall@5 88.64%。",
        "P6：我加入计划完成门禁、一次保守参数修复、只读有限重试、引用白名单、上下文预算、阶段进度事件、取消和并发治理。",
        "P7：我用工作区 confinement 和显式确认交付六个 Word/PDF 工具，并对真实 DOCX/PDF 逐页渲染检查。",
        "P8：我把经典 MiniGPT 改造成可切换 RoPE、RMSNorm、SwiGLU、SDPA 和 fused QKV 的实验平台，同时保持旧 checkpoint 路径兼容。",
        "P8 实验：同参数量下，SDPA 与 fused QKV 的吞吐更高；组合现代配置在小模型、短序列上反而更慢，说明先进组件必须实测。",
        "P8 边界：300 步固定 Batch 只验证实现和局部性能；所有验证 Loss 都恶化，所以我不写成精度提升。",
        "P9：我采用 OCR/ASR/TTS 模块级联，先把上传校验、延迟拆分、失败隔离和 Agent 工具契约做完整，再考虑联合多模态模型。",
        "P9 失败记录：自动 TrOCR processor 在 Transformers 5 环境缺少依赖且推断失败；我补齐 sentencepiece/protobuf，并显式组装图像处理器与 tokenizer 后才完成真实 smoke。",
        "P10：我把私有领域能力设计成默认关闭的 Manifest + 授权 + 数据门禁 + Adapter 激活控制面，公开默认模型不被修改。",
        "成人领域只是可选模板：默认关闭、无相关语料、无权重；启用前必须满足 18+、同意、来源、许可、隐私和禁止内容门禁。",
        "整体验收：Ruff、mypy、split validation 与 256 项测试通过，覆盖率 71.62%；Frozen v1 没有为 P8-P10 重跑。",
    ]
    citations = [
        "docs/p0_p10_closed_loop.md",
        "docs/p8_p10_completion.md",
        "docs/research_log.md",
        "docs/p0_p5_final_frozen_3b_qlora_eval.json",
        "docs/p8_architecture_benchmark.json",
        "docs/p9_multimodal_smoke.json",
        "docs/p10_domain_governance_smoke.json",
    ]
    common = {
        "title": "MiniGPT P0-P10 求职项目验收报告",
        "author": "李林知",
        "summary": (
            "我把项目从教学型 Decoder-only MiniGPT 扩展为具备监督微调、3B QLoRA、"
            "确定性路由、工具执行、RAG、冻结评测、可靠性编排、可观测服务、受控办公文档、"
            "现代架构实验、OCR/ASR/TTS 和私有领域 Adapter 控制面的求职项目。我的原则是"
            "把 Frozen、Dev、固定 Batch benchmark 与单样本 smoke 分开陈述；项目已经形成"
            "完整工程闭环，但不等于生产级通用多模态 Agent。"
        ),
        "metrics": metrics,
        "findings": findings,
        "citations": citations,
        "overwrite": True,
    }
    docx = DocumentWorkspace(ROOT / "output/documents").export_report(
        filename="minigpt_p0_p10_report.docx",
        format="docx",
        **common,
    )
    pdf = DocumentWorkspace(ROOT / "output/pdf").export_report(
        filename="minigpt_p0_p10_report.pdf",
        format="pdf",
        **common,
    )
    print(json.dumps({"docx": docx, "pdf": pdf}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
