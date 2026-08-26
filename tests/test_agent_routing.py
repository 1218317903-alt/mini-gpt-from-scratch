"""Tests for explainable request-level tool routing."""

from agent.messages import Message
from agent.routing import HeuristicAgentRouter


def _route(text: str):
    return HeuristicAgentRouter().route(
        [Message(role="user", content=text)],
        ("calculator", "current_time", "search_docs"),
    )


def test_routes_plain_knowledge_to_direct_answer() -> None:
    decision = _route("水在标准大气压下的沸点是多少？")

    assert decision.kind == "direct"
    assert decision.allowed_tools == ()


def test_routes_explicit_calculation_to_calculator() -> None:
    decision = _route("请计算 37 乘以 19。")

    assert decision.kind == "calculator"
    assert decision.allowed_tools == ("calculator",)


def test_routes_private_document_question_to_search() -> None:
    decision = _route("内部工单系统的域名是什么？")

    assert decision.kind == "retrieval"
    assert decision.allowed_tools == ("search_docs",)


def test_routes_retrieval_then_arithmetic_as_multi_step() -> None:
    decision = _route("先检索年度预算，再计算 36 除以 12。")

    assert decision.kind == "multi_step"
    assert decision.allowed_tools == ("search_docs", "calculator")
