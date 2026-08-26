"""Build deterministic, contamination-checkable Agent SFT trajectories."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.builtin_tools import build_default_registry  # noqa: E402
from agent.llm.tool_format import (  # noqa: E402
    schemas_from_definitions,
    tool_instruction,
)
from evals.metrics import load_cases  # noqa: E402

SEED = 20260825
TIMEZONES = (
    "Asia/Shanghai",
    "Asia/Singapore",
    "Europe/Paris",
    "Europe/Berlin",
    "America/Toronto",
    "America/Los_Angeles",
    "Pacific/Auckland",
)
CONCEPTS = (
    ("梯度累积", "把多个小批次的梯度累加后再更新参数，用较小显存模拟更大的批次。"),
    ("低秩适配", "冻结主模型并训练低秩增量矩阵，以较少参数完成任务适配。"),
    ("检索增强生成", "先检索外部证据，再让模型依据证据生成回答。"),
    ("过拟合", "模型记住训练样本而没有学到可泛化规律，导致验证表现下降。"),
    ("注意力机制", "根据相关性为上下文位置分配权重并聚合信息。"),
    ("量化", "用更低精度表示权重或激活，以降低显存与计算开销。"),
    ("置信区间", "用一个区间表达统计估计的不确定性，而不只给出单点结果。"),
    ("数据污染", "训练数据包含评测答案或高度相似样本，使评测结果虚高。"),
)


def _system_prompt(*tool_names: str) -> str:
    registry = build_default_registry(documents=())
    definitions = tuple(
        definition
        for definition in registry.definitions()
        if definition.name in set(tool_names)
    )
    return tool_instruction(schemas_from_definitions(definitions))


def _prefix(system: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}] if system else []


def _call(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps(
        {"tool_call": {"name": name, "arguments": arguments}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _record(category: str, index: int, messages: list[dict[str, str]]) -> dict:
    return {
        "id": f"agent-sft-{category}-{index:05d}",
        "messages": messages,
        "metadata": {"category": category, "synthetic": True, "seed": SEED},
    }


def _direct(index: int, system: str) -> dict:
    concept, answer = CONCEPTS[index % len(CONCEPTS)]
    limit = 24 + index % 37
    forms = (
        f"请在{limit}字以内解释{concept}，不需要调用工具。",
        f"面试第{index + 1}题：怎样简洁说明{concept}？",
        f"只讲概念，用不超过{limit}字说明什么是{concept}。",
    )
    return _record(
        "direct",
        index,
        _prefix(system)
        + [
            {"role": "user", "content": forms[(index // len(CONCEPTS)) % 3]},
            {"role": "assistant", "content": answer},
        ],
    )


def _calculator(index: int, system: str) -> dict:
    left = 113 + (index * 37) % 887
    right = 7 + (index * 19) % 91
    operation = index % 3
    if operation == 0:
        expression, value, wording = f"{left}+{right}", left + right, "加上"
    elif operation == 1:
        expression, value, wording = f"{left}-{right}", left - right, "减去"
    else:
        expression, value, wording = f"{left}*{right}", left * right, "乘以"
    question_forms = (
        f"请用计算器精确计算 {left} {wording} {right}。",
        f"算出 {left}{'+' if operation == 0 else '-' if operation == 1 else '×'}{right}，请调用工具避免心算误差。",
        f"我需要 {left} {wording} {right} 的精确值，请交给计算器。",
        f"做一次数值核验：{expression} 等于多少？",
    )
    return _record(
        "calculator",
        index,
        _prefix(system)
        + [
            {
                "role": "user",
                "content": question_forms[index % len(question_forms)],
            },
            {
                "role": "assistant",
                "content": _call("calculator", {"expression": expression}),
            },
            {"role": "tool", "name": "calculator", "content": str(value)},
            {"role": "assistant", "content": f"计算结果是 {value}。"},
        ],
    )


def _time(index: int, system: str) -> dict:
    timezone = TIMEZONES[index % len(TIMEZONES)]
    forms = (
        f"请查询 {timezone} 的当前时间，并标明时区；请求编号 T{index:04d}。",
        f"现在 {timezone} 是几点？这是时区检查 T{index:04d}。",
        f"调用时间工具获取 {timezone} 当前时刻，记录号 T{index:04d}。",
        f"帮我核对 {timezone} 的本地时间，任务 T{index:04d}。",
    )
    return _record(
        "current_time",
        index,
        _prefix(system)
        + [
            {"role": "user", "content": forms[index % len(forms)]},
            {
                "role": "assistant",
                "content": _call("current_time", {"timezone": timezone}),
            },
            {
                "role": "tool",
                "name": "current_time",
                "content": f"2026-08-{10 + index % 15:02d}T{8 + index % 10:02d}:30:00+08:00",
            },
            {"role": "assistant", "content": f"工具返回的 {timezone} 当前时间如上。"},
        ],
    )


def _retrieval(index: int, system: str) -> dict:
    project = f"星屿项目{index:04d}"
    owner = f"研发组{index % 17 + 1}"
    citation = f"doc-train-{index:05d}"
    query = f"{project} 负责人"
    result = {
        "query": query,
        "results": [
            {
                "citation": citation,
                "text": f"{project}的负责人是{owner}。",
                "score": 1.0,
            }
        ],
    }
    forms = (
        f"请查内部资料：{project}由哪个团队负责？",
        f"内部文档中，{project}的责任团队是谁？",
        f"帮我检索{project}负责人，答案要带引用。",
        f"知识库里记录的{project}归属团队是什么？",
        f"只依据资料回答：谁负责{project}？",
        f"搜索项目文档，确认{project}的负责小组。",
        f"我想查{project}的内部负责人，请附证据编号。",
        f"从文档中找到{project}由哪个研发组维护。",
    )
    return _record(
        "retrieval",
        index,
        _prefix(system)
        + [
            {"role": "user", "content": forms[index % len(forms)]},
            {
                "role": "assistant",
                "content": _call("search_docs", {"query": query, "k": 3}),
            },
            {
                "role": "tool",
                "name": "search_docs",
                "content": json.dumps(
                    result, ensure_ascii=False, separators=(",", ":")
                ),
            },
            {
                "role": "assistant",
                "content": f"根据资料，{project}由{owner}负责（{citation}）。",
            },
        ],
    )


def _recovery(index: int, system: str) -> dict:
    left = 211 + index
    right = 11 + index % 89
    expression = f"{left}*{right}"
    value = left * right
    call = _call("calculator", {"expression": expression})
    forms = (
        f"用计算器算 {left} 乘以 {right}；若临时失败请重试。",
        f"精确核算 {left}×{right}，工具超时后允许再试一次。",
        f"请调用计算器求 {expression}；遇到瞬时错误不要直接放弃。",
    )
    return _record(
        "recovery",
        index,
        _prefix(system)
        + [
            {"role": "user", "content": forms[index % len(forms)]},
            {"role": "assistant", "content": call},
            {
                "role": "tool",
                "name": "calculator",
                "content": "Tool timed out. This transient call may be retried once.",
            },
            {"role": "assistant", "content": call},
            {"role": "tool", "name": "calculator", "content": str(value)},
            {"role": "assistant", "content": f"重试成功，结果是 {value}。"},
        ],
    )


BUILDERS = (_direct, _calculator, _time, _retrieval, _recovery)


def build_records(count: int) -> list[dict]:
    systems = (
        "",
        _system_prompt("calculator"),
        _system_prompt("current_time"),
        _system_prompt("search_docs"),
        _system_prompt("calculator"),
    )
    records = [
        BUILDERS[index % len(BUILDERS)](index, systems[index % len(systems)])
        for index in range(count)
    ]
    random.Random(SEED).shuffle(records)
    return records


def _normalized_user(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())


def _character_ngrams(text: str, width: int = 3) -> set[str]:
    if len(text) < width:
        return {text} if text else set()
    return {text[index : index + width] for index in range(len(text) - width + 1)}


def contamination_report(records: list[dict], cases_dirs: list[Path]) -> dict:
    eval_prompts: dict[str, str] = {}
    for cases_dir in cases_dirs:
        for case in load_cases(cases_dir):
            user = next(
                message["content"]
                for message in reversed(case.messages)
                if message["role"] == "user"
            )
            eval_prompts[case.id] = _normalized_user(user)
    train_prompts = {
        record["id"]: _normalized_user(
            next(
                message["content"]
                for message in record["messages"]
                if message["role"] == "user"
            )
        )
        for record in records
    }
    prompt_counts = Counter(train_prompts.values())
    internal_duplicates = sorted(
        prompt for prompt, count in prompt_counts.items() if count > 1
    )
    exact = [
        {"record_id": record_id, "case_id": case_id}
        for record_id, prompt in train_prompts.items()
        for case_id, eval_prompt in eval_prompts.items()
        if prompt == eval_prompt
    ]
    eval_ngrams = {
        case_id: _character_ngrams(prompt) for case_id, prompt in eval_prompts.items()
    }
    near: list[dict[str, Any]] = []
    maximum_similarity = 0.0
    for record_id, prompt in train_prompts.items():
        grams = _character_ngrams(prompt)
        for case_id, comparison in eval_ngrams.items():
            union = grams | comparison
            similarity = len(grams & comparison) / len(union) if union else 1.0
            maximum_similarity = max(maximum_similarity, similarity)
            if similarity >= 0.8 and prompt != eval_prompts[case_id]:
                near.append(
                    {
                        "record_id": record_id,
                        "case_id": case_id,
                        "character_trigram_jaccard": round(similarity, 6),
                    }
                )
    return {
        "training_records": len(records),
        "unique_training_prompts": len(prompt_counts),
        "internal_duplicate_prompt_count": len(internal_duplicates),
        "evaluation_prompts": len(eval_prompts),
        "normalization": "lowercase alphanumeric characters",
        "exact_overlap_count": len(exact),
        "exact_overlaps": exact,
        "near_overlap_threshold": 0.8,
        "near_overlap_count": len(near),
        "near_overlaps": near,
        "maximum_character_trigram_jaccard": round(maximum_similarity, 6),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("hf_lora/data/agent_v1")
    )
    parser.add_argument("--train-count", type=int, default=2400)
    parser.add_argument("--validation-count", type=int, default=300)
    args = parser.parse_args(argv)
    records = build_records(args.train_count + args.validation_count)
    train = records[: args.train_count]
    validation = records[args.train_count :]
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    report = contamination_report(
        records,
        [Path("evals/cases"), Path("evals/frozen_cases")],
    )
    report["category_counts"] = dict(
        sorted(Counter(record["metadata"]["category"] for record in records).items())
    )
    report["train_count"] = len(train)
    report["validation_count"] = len(validation)
    (args.output_dir / "contamination_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return (
        0
        if (
            not report["exact_overlap_count"]
            and not report["near_overlap_count"]
            and not report["internal_duplicate_prompt_count"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
