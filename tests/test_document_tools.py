"""P7 path safety, Word/PDF creation, extraction, export, and routing tests."""

from pathlib import Path

import pytest

from agent.builtin_tools import build_default_registry
from agent.document_tools import DocumentWorkspace
from agent.messages import Message
from agent.routing import HeuristicAgentRouter
from agent.tools.base import ToolCall, tool_confirmation_id
from agent.tools.executor import ToolExecutor


@pytest.fixture
def workspace(tmp_path: Path) -> DocumentWorkspace:
    return DocumentWorkspace(tmp_path / "documents")


def _payload(filename: str) -> dict[str, object]:
    return {
        "filename": filename,
        "title": "P7 文档工具验收",
        "subtitle": "结构、表格、页码与引用",
        "author": "李林知",
        "sections": [
            {"heading": "执行摘要", "body": "文档工具在受控目录中生成文件。"},
            {"heading": "验收结论", "bullets": ["结构校验通过", "重新读取通过"]},
        ],
        "metrics": [{"label": "测试", "value": "通过"}],
        "citations": ["docs/p0_p5_closed_loop.md"],
    }


@pytest.mark.parametrize(
    "filename",
    ["../escape.docx", "folder/file.docx", "folder\\file.docx", "bad?.docx"],
)
def test_document_workspace_rejects_unsafe_filenames(
    workspace: DocumentWorkspace,
    filename: str,
) -> None:
    with pytest.raises(ValueError):
        workspace.create_docx(**_payload(filename))


@pytest.mark.parametrize(
    ("method", "filename"),
    [("create_docx", "wrong.pdf"), ("create_pdf", "wrong.docx")],
)
def test_creation_requires_matching_plain_suffix(
    workspace: DocumentWorkspace,
    method: str,
    filename: str,
) -> None:
    with pytest.raises(ValueError, match="plain"):
        getattr(workspace, method)(**_payload(filename))


def test_create_and_read_docx_with_table_and_citations(
    workspace: DocumentWorkspace,
) -> None:
    created = workspace.create_docx(**_payload("report.docx"))
    extracted = workspace.read_docx("report.docx")

    assert created["validated"] is True
    assert created["tables"] == 1
    assert "P7 文档工具验收" in extracted["text"]
    assert "docs/p0_p5_closed_loop.md" in extracted["text"]


def test_docx_requires_explicit_overwrite(workspace: DocumentWorkspace) -> None:
    workspace.create_docx(**_payload("same.docx"))

    with pytest.raises(ValueError, match="overwrite=true"):
        workspace.create_docx(**_payload("same.docx"))

    replaced = workspace.create_docx(**_payload("same.docx"), overwrite=True)
    assert replaced["validated"] is True


def test_create_and_read_pdf_with_page_count(workspace: DocumentWorkspace) -> None:
    created = workspace.create_pdf(**_payload("report.pdf"))
    extracted = workspace.read_pdf("report.pdf")

    assert created["validated"] is True
    assert extracted["pages"] >= 1
    assert "P7 文档工具验收" in extracted["text"]


@pytest.mark.parametrize("suffix", ["docx", "pdf"])
def test_summarize_document_is_extractive(
    workspace: DocumentWorkspace,
    suffix: str,
) -> None:
    filename = f"summary.{suffix}"
    getattr(workspace, f"create_{suffix}")(**_payload(filename))

    result = workspace.summarize_document(filename, summary_characters=200)

    assert result["extractive"] is True
    assert "P7 文档工具验收" in result["summary"]


def test_summarize_rejects_unknown_extension(workspace: DocumentWorkspace) -> None:
    with pytest.raises(ValueError, match="only .docx and .pdf"):
        workspace.summarize_document("notes.txt")


@pytest.mark.parametrize("format", ["docx", "pdf"])
def test_export_report_round_trip(
    workspace: DocumentWorkspace,
    format: str,
) -> None:
    filename = f"export.{format}"
    result = workspace.export_report(
        filename=filename,
        format=format,
        title="P6/P7 阶段报告",
        summary="可靠性和文档能力已经形成闭环。",
        metrics=[{"label": "测试数", "value": "20+"}],
        findings=["路径安全有效", "结构化回读有效"],
        citations=["docs/research_log.md"],
    )

    assert result["validated"] is True
    assert (workspace.root / filename).is_file()


def test_read_rejects_missing_document(workspace: DocumentWorkspace) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        workspace.read_docx("missing.docx")


def test_registry_adds_exactly_six_document_tools(tmp_path: Path) -> None:
    registry = build_default_registry(document_workspace=tmp_path)

    assert {
        "create_docx",
        "create_pdf",
        "read_docx",
        "read_pdf",
        "summarize_document",
        "export_report",
    }.issubset(registry.names())


def test_document_writes_require_confirmation(tmp_path: Path) -> None:
    registry = build_default_registry(document_workspace=tmp_path)
    executor = ToolExecutor(registry)
    call = ToolCall("write", "create_docx", _payload("agent.docx"))
    try:
        blocked = executor.execute(call)
        approved = executor.execute(
            call,
            approved_side_effects={tool_confirmation_id(call)},
        )
    finally:
        executor.close()

    assert blocked.status == "confirmation_required"
    assert approved.status == "ok"
    assert (tmp_path / "agent.docx").is_file()


def test_document_reads_do_not_require_confirmation(
    workspace: DocumentWorkspace,
) -> None:
    workspace.create_docx(**_payload("read.docx"))
    registry = build_default_registry(document_workspace=workspace.root)
    executor = ToolExecutor(registry)
    try:
        result = executor.execute(
            ToolCall("read", "read_docx", {"filename": "read.docx"})
        )
    finally:
        executor.close()

    assert result.status == "ok"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("请生成一份 Word 文档", "create_docx"),
        ("请制作一个 PDF 文件", "create_pdf"),
        ("读取这个 PDF", "read_pdf"),
        ("总结文档内容", "summarize_document"),
        ("导出一份项目报告", "export_report"),
    ],
)
def test_router_selects_document_capability(text: str, expected: str) -> None:
    available = (
        "create_docx",
        "create_pdf",
        "read_docx",
        "read_pdf",
        "summarize_document",
        "export_report",
    )

    decision = HeuristicAgentRouter().route(
        [Message(role="user", content=text)], available
    )

    assert decision.kind == "document"
    assert decision.allowed_tools == (expected,)
