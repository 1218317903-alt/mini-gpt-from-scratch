"""Explainable first-stage routing for the bounded text Agent."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.messages import Message, as_message

RouteKind = Literal[
    "direct",
    "calculator",
    "current_time",
    "retrieval",
    "multi_step",
    "document",
    "multimodal",
    "auto",
]


@dataclass(frozen=True)
class RouteDecision:
    """One auditable decision about which tools the model may see."""

    kind: RouteKind
    allowed_tools: tuple[str, ...]
    reason: str


class AgentRouter(Protocol):
    """Contract for request-level tool routing."""

    def route(
        self,
        messages: Sequence[Message | object],
        available_tools: Sequence[str],
    ) -> RouteDecision:
        """Select the model-visible subset of registered tools."""


class HeuristicAgentRouter:
    """Small deterministic baseline designed for inspection and ablation."""

    _RETRIEVAL_MARKERS = (
        "根据资料",
        "资料里",
        "查资料",
        "文档",
        "知识库",
        "检索",
        "搜索",
        "青梧实验室",
        "澄川研究所",
        "查到",
        "值班分机",
        "算力预算",
        "本地实用主模型",
        "内部工单",
        "访客 wi-fi",
        "访客 wifi",
        "入职当年",
        "数据集备份",
        "评测集能否",
        "工具白名单",
        "产品周会",
        "打印机的驱动",
    )
    _DIRECT_MARKERS = (
        "解释",
        "什么是",
        "为什么",
        "区别",
        "含义",
        "原理",
        "不要计算",
        "不要求计算",
        "只做概念",
        "直接判断",
        "主要用来",
        "主要解决",
        "什么情况下",
        "有什么不同",
        "一般表示什么",
        "数量级估计",
        "运算关系",
    )
    _TIME_MARKERS = (
        "当前时间",
        "现在几点",
        "现在的utc",
        "现在的 utc",
        "当前本地时间",
        "当前时刻",
        "时间戳",
        "查询时间",
        "协调世界时",
        "时区",
    )
    _CALCULATOR_MARKERS = (
        "请计算",
        "精确计算",
        "用工具计算",
        "用计算器",
        "算出",
        "平方根",
        "次方",
        "乘以",
        "除以",
        "百分之",
        "连加",
        "计算直径",
        "再计算",
        "计算器",
        "减去",
        "加上",
    )
    _DOCUMENT_CREATE_MARKERS = ("生成", "创建", "制作", "导出", "写一份")
    _DOCUMENT_READ_MARKERS = ("读取", "打开", "提取", "解析")
    _DOCUMENT_SUMMARY_MARKERS = ("总结文档", "总结文件", "概括文档", "摘要")

    def route(
        self,
        messages: Sequence[Message | object],
        available_tools: Sequence[str],
    ) -> RouteDecision:
        normalized = [as_message(message) for message in messages]
        text = next(
            (
                message.content.strip().lower()
                for message in reversed(normalized)
                if message.role == "user"
            ),
            "",
        )
        available = tuple(sorted(set(available_tools)))
        retrieval = any(marker in text for marker in self._RETRIEVAL_MARKERS)
        direct = any(marker in text for marker in self._DIRECT_MARKERS)
        current_time = self._needs_current_time(text)
        calculator = self._needs_calculator(text)
        document_tools = self._document_tools(text, available)
        multimodal_tools = self._multimodal_tools(text, available)

        requested: list[str] = []
        if retrieval:
            requested.append("search_docs")
        if calculator:
            requested.append("calculator")
        if current_time:
            requested.append("current_time")
        requested.extend(document_tools)
        requested.extend(multimodal_tools)
        allowed = tuple(name for name in requested if name in available)

        if len(allowed) >= 2:
            return RouteDecision(
                "multi_step",
                allowed,
                "The request contains signals for multiple explicit tool capabilities.",
            )
        if retrieval and "search_docs" in allowed:
            return RouteDecision(
                "retrieval",
                allowed,
                "The request refers to supplied documents or a named private corpus.",
            )
        if direct:
            return RouteDecision(
                "direct",
                (),
                "The request asks for explanation or judgment rather than live data.",
            )
        if calculator and "calculator" in allowed:
            return RouteDecision(
                "calculator",
                allowed,
                "The request explicitly asks for a numeric calculation.",
            )
        if current_time and "current_time" in allowed:
            return RouteDecision(
                "current_time",
                allowed,
                "The request asks for the current time in a concrete timezone.",
            )
        if document_tools and allowed:
            return RouteDecision(
                "document",
                allowed,
                "The request explicitly asks to create, read, summarize, or export a document.",
            )
        if multimodal_tools and allowed:
            return RouteDecision(
                "multimodal",
                allowed,
                "The request explicitly asks to analyze an image or process speech.",
            )
        return RouteDecision(
            "direct",
            (),
            "No explicit tool signal matched; answer without exposing tools.",
        )

    def _needs_current_time(self, text: str) -> bool:
        if any(marker in text for marker in self._TIME_MARKERS):
            return True
        timezone_name = re.search(
            r"\b(?:asia|america|europe|africa|pacific)/[a-z_]+", text
        )
        return timezone_name is not None and any(
            marker in text for marker in ("时间", "几点", "时刻")
        )

    def _needs_calculator(self, text: str) -> bool:
        if not re.search(r"\d", text):
            return False
        if any(marker in text for marker in self._CALCULATOR_MARKERS):
            return True
        return re.search(r"\d\s*(?:[+\-*/×÷]|加|减|乘|除)\s*\d", text) is not None

    def _document_tools(
        self,
        text: str,
        available_tools: Sequence[str],
    ) -> tuple[str, ...]:
        available = set(available_tools)
        is_docx = any(marker in text for marker in ("word", "docx", "文档"))
        is_pdf = "pdf" in text
        if any(marker in text for marker in self._DOCUMENT_SUMMARY_MARKERS):
            return ("summarize_document",) if "summarize_document" in available else ()
        if any(marker in text for marker in self._DOCUMENT_READ_MARKERS):
            name = "read_pdf" if is_pdf else "read_docx" if is_docx else ""
            return (name,) if name in available else ()
        if "报告" in text and any(
            marker in text for marker in self._DOCUMENT_CREATE_MARKERS
        ):
            return ("export_report",) if "export_report" in available else ()
        if any(marker in text for marker in self._DOCUMENT_CREATE_MARKERS):
            name = "create_pdf" if is_pdf else "create_docx" if is_docx else ""
            return (name,) if name in available else ()
        return ()

    @staticmethod
    def _multimodal_tools(text: str, available_tools: Sequence[str]) -> tuple[str, ...]:
        available = set(available_tools)
        requested: list[str] = []
        if any(marker in text for marker in ("图片", "图像", "截图", "ocr", "扫描件")):
            requested.append("analyze_image")
        if any(marker in text for marker in ("语音转文字", "转录音频", "asr", "听写")):
            requested.append("transcribe_audio")
        if any(marker in text for marker in ("文字转语音", "语音回答", "tts", "朗读")):
            requested.append("synthesize_speech")
        return tuple(name for name in requested if name in available)
