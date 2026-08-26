"""Export the reproducible P0-P7 project report as DOCX and PDF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.document_tools import DocumentWorkspace

ROOT = Path(__file__).resolve().parents[1]


def _load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["scoring"]["metrics"]


def main() -> None:
    """Create both artifacts through the same validated document contract."""
    frozen = _load_metrics(ROOT / "docs/p0_p5_final_frozen_3b_qlora_eval.json")
    p6 = _load_metrics(ROOT / "docs/p6_agent_v1_1_3b_qlora_dev_eval.json")
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
        {
            "label": "P6 Dev Multi-step 成功率",
            "value": f"{14 / 15:.2%} (14/15)",
        },
        {
            "label": "P6 Dev 不必要工具调用率",
            "value": f"{p6['unnecessary_tool_rate']:.2%}",
        },
        {"label": "全量工程测试", "value": "227 passed"},
        {"label": "代码覆盖率", "value": "71.36% (门槛 60%)"},
        {"label": "P7 文档工具", "value": "6 个受控工具"},
    ]
    findings = [
        "我把 multi-step 从路由标签升级为计划完成门禁；Dev 成功率由 86.67% 提高到 93.33%。",
        "我增加一次保守参数修复、只读工具瞬时失败有限重试、协作式取消和上下文预算。",
        "我用本次 search_docs 返回值建立引用白名单；它能阻止伪引用，但不能自动补齐漏引用。",
        "服务新增 SSE 进度事件、请求 ID、并发上限、取消接口、聚合指标和本地可观测 Demo。",
        "Word/PDF 写操作必须显式确认，路径限定在工作区，拒绝目录穿越和静默覆盖。",
        "DOCX/PDF 创建后都会重新读取关键字段；最终示例还执行页面渲染和逐页视觉检查。",
    ]
    citations = [
        "docs/p0_p5_closed_loop.md",
        "docs/p0_p5_final_frozen_3b_qlora_eval.json",
        "docs/p6_agent_v1_1_3b_qlora_dev_eval.json",
        "docs/p6_3b_before_vs_reliability_v2.json",
        "docs/research_log.md",
    ]
    common = {
        "title": "MiniGPT P0-P7 求职项目验收报告",
        "author": "李林知",
        "summary": (
            "我已经把项目从教学型 Decoder-only MiniGPT 扩展为具备监督微调、QLoRA、"
            "确定性路由、工具执行、RAG、冻结评测、可观测服务和受控办公文档能力的"
            "纯文本 Agent。P6/P7 的价值不只是增加接口，而是把冻结测试暴露的多步骤、"
            "参数、恢复、引用和实际文件交付问题转化为可测试的系统约束。"
        ),
        "metrics": metrics,
        "findings": findings,
        "citations": citations,
        "overwrite": True,
    }
    docx = DocumentWorkspace(ROOT / "output/documents").export_report(
        filename="minigpt_p0_p7_report.docx",
        format="docx",
        **common,
    )
    pdf = DocumentWorkspace(ROOT / "output/pdf").export_report(
        filename="minigpt_p0_p7_report.pdf",
        format="pdf",
        **common,
    )
    print(json.dumps({"docx": docx, "pdf": pdf}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
