"""Deterministically build the frozen-test candidate without network access."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.inventory import build_manifest_payload, write_manifest  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "frozen_cases"


def base_case(
    case_id: str,
    category: str,
    question: str,
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": case_id,
        "category": category,
        "messages": [{"role": "user", "content": question}],
        "expect_tool": False,
        "expected_tools": [],
        "forbidden_tools": ["calculator", "current_time", "search_docs"],
        "expected_arguments": None,
        "gold_answer": None,
        "answer_contains": [],
        "success_mode": "answer_contains",
        "citations": [],
        "retrieval": None,
        "max_steps": 6,
        "min_tool_calls": 0,
        "expect_recovery": False,
        "subcategory": None,
        "notes": "frozen-test candidate; do not use for prompt or SFT tuning",
    }
    payload.update(overrides)
    return payload


def calculator_case(
    case_id: str,
    category: str,
    question: str,
    expression: str,
    answer: str,
) -> dict[str, Any]:
    return base_case(
        case_id,
        category,
        question,
        expect_tool=True,
        expected_tools=["calculator"],
        forbidden_tools=["current_time", "search_docs"],
        expected_arguments={"calculator": {"expression": expression}},
        gold_answer=answer,
        answer_contains=[answer],
        success_mode="tool_and_answer",
    )


def time_case(
    case_id: str,
    category: str,
    question: str,
    timezone: str,
) -> dict[str, Any]:
    return base_case(
        case_id,
        category,
        question,
        expect_tool=True,
        expected_tools=["current_time"],
        forbidden_tools=["calculator", "search_docs"],
        expected_arguments={"current_time": {"timezone": timezone}},
        success_mode="tool_only",
    )


def write_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
    (OUTPUT / name).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def direct_cases() -> list[dict[str, Any]]:
    facts = [
        ("人体负责泵送血液的主要器官是什么？", "心脏"),
        ("植物进行光合作用时通常吸收哪种气体？", "二氧化碳"),
        ("圆周率常用的希腊字母符号是什么？", "π"),
        ("中国传统二十四节气中，春季的第一个节气是什么？", "立春"),
        ("电子邮件地址中连接用户名和域名的符号是什么？", "@"),
        ("世界上面积最大的海洋是哪一个？", "太平洋"),
        ("金的化学元素符号是什么？", "Au"),
        ("Python 中用于定义函数的关键字是什么？", "def"),
        ("联合国总部位于哪座城市？", "纽约"),
        ("摄氏零度时，纯水通常处于哪种相变温度？", "冰点"),
        ("《西游记》中护送唐僧西行的大师兄是谁？", "孙悟空"),
        ("计算机 CPU 的中文常用名称是什么？", "中央处理器"),
        ("太阳从地理方位上通常在哪一侧升起？", "东"),
        ("英语字母表共有多少个字母？", "26"),
        ("声音在真空中能否传播？", "不能"),
        ("Linux 中表示当前目录的单个点号是什么？", "."),
        ("地球唯一的天然卫星叫什么？", "月球"),
        ("国际单位制中质量的基本单位是什么？", "千克"),
        ("HTTP 状态码 404 通常表示什么？", "未找到"),
        ("汉字“森”由几个“木”组成？", "3"),
        ("水分子的化学式是什么？", "H2O"),
        ("Git 中查看工作区状态的常用子命令是什么？", "status"),
        ("珠穆朗玛峰位于喜马拉雅山脉吗？", "是"),
        ("十进制数字 10 用二进制表示是什么？", "1010"),
        ("一年四季中位于春季之后的是哪个季节？", "夏"),
        ("键盘快捷键 Ctrl+C 通常表示什么操作？", "复制"),
        ("HTML 中创建超链接通常使用哪个标签？", "a"),
        ("三原色中的红色、蓝色之外，另一种通常是什么？", "黄"),
        ("地球表面大部分被陆地还是海洋覆盖？", "海洋"),
        ("数据库事务 ACID 中的 A 通常代表什么？", "原子性"),
    ]
    return [
        base_case(
            f"fda-{index:03d}",
            "direct_answer",
            question,
            gold_answer=answer,
            answer_contains=[answer],
        )
        for index, (question, answer) in enumerate(facts, start=1)
    ]


def tool_selection_cases() -> list[dict[str, Any]]:
    calculations = [
        ("请精确计算 46 乘以 19。", "46*19", "874"),
        ("请用计算器求 2048 除以 32。", "2048/32", "64"),
        ("计算 (73-25) 乘 3。", "(73-25)*3", "144"),
        ("请计算 5 的 6 次方。", "5**6", "15625"),
        ("精确计算 12.5 加 7.75。", "12.5+7.75", "20.25"),
        ("请算出 900 减 468。", "900-468", "432"),
        ("计算 144 的平方根。", "144**0.5", "12"),
        ("请计算 17 乘 18 再减 6。", "17*18-6", "300"),
        ("求 3 除以 16 的小数结果。", "3/16", "0.1875"),
        ("请精确计算 2.4 乘 7.5。", "2.4*7.5", "18"),
        ("计算 (8+4) 的平方。", "(8+4)**2", "144"),
        ("请计算 72 的四分之三。", "72*0.75", "54"),
        ("精确计算 1001 加 999。", "1001+999", "2000"),
        ("计算 -13 加 29。", "-13+29", "16"),
        ("请计算 6 乘 7 乘 8。", "6*7*8", "336"),
        ("求 (120-20)/4。", "(120-20)/4", "25"),
        ("计算 0.125 乘 64。", "0.125*64", "8"),
        ("请计算 11 的立方。", "11**3", "1331"),
        ("精确计算 250 的 18%。", "250*0.18", "45"),
        ("请算 7.2 除以 0.9。", "7.2/0.9", "8"),
        ("计算 (15+5)*(9-4)。", "(15+5)*(9-4)", "100"),
        ("请精确计算 4096 除以 8。", "4096/8", "512"),
        ("计算 33 乘 33。", "33*33", "1089"),
        ("请计算 1.5 加 2.25 再加 3.75。", "1.5+2.25+3.75", "7.5"),
    ]
    tools = [
        calculator_case(
            f"fts-{index:03d}",
            "tool_selection",
            question,
            expression,
            answer,
        )
        for index, (question, expression, answer) in enumerate(calculations, start=1)
    ]
    times = [
        ("请查询首尔当前本地时间。", "Asia/Seoul"),
        ("现在的 Europe/Berlin 时间是什么？", "Europe/Berlin"),
        ("请给出 Pacific/Auckland 的当前时间。", "Pacific/Auckland"),
        ("查询 Asia/Kolkata 现在几点。", "Asia/Kolkata"),
        ("请读取 America/Chicago 的当前时间。", "America/Chicago"),
        ("现在的 Africa/Cairo 时间是什么？", "Africa/Cairo"),
    ]
    tools.extend(
        time_case(
            f"fts-{index:03d}",
            "tool_selection",
            question,
            timezone,
        )
        for index, (question, timezone) in enumerate(times, start=25)
    )
    distractors = [
        "请解释平方根的含义，不要求计算任何具体数字。",
        "为什么不同国家会采用不同的标准时间？",
        "计算器的存储键通常有什么用途？",
        "请说明指数运算和普通乘法的区别。",
        "什么是世界协调时？只解释概念。",
        "为什么浮点数可能出现舍入误差？",
        "时区夏令时会给软件系统带来什么问题？",
        "请描述括号在数学表达式中的作用。",
        "心算和使用计算器各自适合什么场景？",
        "IANA 时区名称为什么通常包含斜杠？",
        "请解释百分数与小数之间的关系。",
        "乘方运算中的底数和指数分别是什么？",
        "什么是负数？请给出概念说明。",
        "原子钟为什么比普通机械钟准确？",
        "请说明除法中的被除数和除数。",
    ]
    tools.extend(
        base_case(
            f"fts-d-{index:03d}",
            "tool_selection",
            question,
            subcategory="distractor",
        )
        for index, question in enumerate(distractors, start=1)
    )
    return tools


def tool_argument_cases() -> list[dict[str, Any]]:
    calculations = [
        ("去掉千分位后计算 2,500 加 75。", "2500+75", "2575"),
        ("用 ** 表示指数，计算 3 的 7 次方。", "3**7", "2187"),
        ("保持括号，计算 50 除以 5 加 5。", "50/(5+5)", "5"),
        ("计算负 12 乘以 4。", "-12*4", "-48"),
        ("用 0.2 表示百分之二十，计算 85 的百分之二十。", "85*0.2", "17"),
        ("计算 7 除以 8 的小数值。", "7/8", "0.875"),
        ("先算 9 加 3，再整体乘 5。", "(9+3)*5", "60"),
        ("计算 16 的负一次方。", "16**-1", "0.0625"),
        ("将 6.4 与 1.25 相乘。", "6.4*1.25", "8"),
        ("计算 9999 减 1111。", "9999-1111", "8888"),
        ("计算 (2+6)/(3+1)。", "(2+6)/(3+1)", "2"),
        ("使用小数 0.375 计算 80 的 37.5%。", "80*0.375", "30"),
        ("计算 121 的平方根，用幂表达式。", "121**0.5", "11"),
        ("将 14 连续相加 6 次，用乘法表达。", "14*6", "84"),
        ("计算 0.04 加 0.006。", "0.04+0.006", "0.046"),
        ("先计算 100 减 64，再除以 9。", "(100-64)/9", "4"),
        ("计算 2 的 5 次方再加 1。", "2**5+1", "33"),
        ("用 3.1416 计算直径为 10 的圆周长。", "3.1416*10", "31.416"),
    ]
    rows = [
        calculator_case(
            f"fta-{index:03d}",
            "tool_arguments",
            question,
            expression,
            answer,
        )
        for index, (question, expression, answer) in enumerate(calculations, start=1)
    ]
    times = [
        ("查询温哥华时间，参数必须使用 America/Vancouver。", "America/Vancouver"),
        ("查询迪拜时间，使用 Asia/Dubai。", "Asia/Dubai"),
        ("查询罗马时间，使用 Europe/Rome。", "Europe/Rome"),
        ("查询檀香山时间，使用 Pacific/Honolulu。", "Pacific/Honolulu"),
        ("查询新加坡时间，使用 Asia/Singapore。", "Asia/Singapore"),
        ("查询马德里时间，使用 Europe/Madrid。", "Europe/Madrid"),
        (
            "查询布宜诺斯艾利斯时间，使用 America/Argentina/Buenos_Aires。",
            "America/Argentina/Buenos_Aires",
        ),
    ]
    rows.extend(
        time_case(
            f"fta-{index:03d}",
            "tool_arguments",
            question,
            timezone,
        )
        for index, (question, timezone) in enumerate(times, start=19)
    )
    return rows


DOCUMENT_SPECS = [
    (
        "lab-address",
        "澄川研究所的访客入口位于苏州星湖街 88 号北门。",
        "访客入口位于哪条街？",
        "星湖街",
        "访客入口 星湖街 88号",
    ),
    (
        "lab-hours",
        "澄川研究所工作日开放时间为 08:45 至 17:45。",
        "工作日几点停止开放？",
        "17:45",
        "工作日 开放 17:45",
    ),
    (
        "lab-gpu",
        "澄川研究所的离线评测机配备一张 RTX 4070 Ti SUPER，显存 16GB。",
        "离线评测机有多少显存？",
        "16GB",
        "离线评测机 显存 16GB",
    ),
    (
        "lab-model",
        "澄川研究所当前文本基线模型代号为 RiverLM-3B。",
        "当前文本基线模型代号是什么？",
        "RiverLM-3B",
        "文本基线 RiverLM-3B",
    ),
    (
        "lab-backup",
        "向量索引每周六 01:30 执行完整备份。",
        "向量索引完整备份在周几？",
        "周六",
        "向量索引 完整备份 周六",
    ),
    (
        "lab-retention",
        "脱敏推理日志默认保留 14 天，之后自动删除。",
        "脱敏推理日志保留多少天？",
        "14",
        "脱敏 推理日志 14天",
    ),
    (
        "lab-review",
        "冻结测试集发布前需要数据负责人和评测负责人双人签字。",
        "冻结测试集发布前需要几类负责人签字？",
        "双人",
        "冻结测试集 双人 签字",
    ),
    (
        "lab-wifi",
        "访客网络名称为 Chengchuan-Guest，仅前台可发放临时密码。",
        "访客网络名称是什么？",
        "Chengchuan-Guest",
        "访客网络 Chengchuan-Guest",
    ),
    (
        "lab-room",
        "模型复盘会固定在周四 15:30，于 C4 会议室举行。",
        "模型复盘会在哪个会议室？",
        "C4",
        "模型复盘会 C4",
    ),
    (
        "lab-owner",
        "评测平台的值班负责人是顾清，内部短号为 7312。",
        "评测平台值班短号是多少？",
        "7312",
        "评测平台 值班 短号 7312",
    ),
    (
        "lab-ticket",
        "数据问题通过 data-ticket.chengchuan.local 提交。",
        "数据问题提交域名是什么？",
        "data-ticket.chengchuan.local",
        "数据问题 data-ticket.chengchuan.local",
    ),
    (
        "lab-citation",
        "知识库回答必须使用方括号标注文档编号。",
        "知识库回答用什么符号标注文档编号？",
        "方括号",
        "知识库回答 方括号 文档编号",
    ),
    (
        "lab-topk",
        "当前词法检索默认返回 Top-4，向量检索默认返回 Top-8。",
        "当前词法检索默认返回几条？",
        "4",
        "词法检索 Top-4",
    ),
    (
        "lab-timeout",
        "只读检索工具的默认超时时间为 3 秒。",
        "只读检索工具默认超时多少秒？",
        "3",
        "只读检索 默认超时 3秒",
    ),
    (
        "lab-upload",
        "单个文本知识文件最大允许 20MB。",
        "单个文本知识文件最大允许多大？",
        "20MB",
        "文本知识文件 最大 20MB",
    ),
    (
        "lab-chunk",
        "中文技术文档切分默认目标长度为 480 个字符。",
        "中文技术文档默认切分目标是多少字符？",
        "480",
        "中文技术文档 切分 480字符",
    ),
    (
        "lab-overlap",
        "相邻文档块默认重叠 60 个字符。",
        "相邻文档块默认重叠多少字符？",
        "60",
        "文档块 重叠 60字符",
    ),
    (
        "lab-index",
        "生产索引版本命名格式为 knowledge-YYYYMMDD。",
        "生产索引使用什么命名格式？",
        "knowledge-YYYYMMDD",
        "生产索引 knowledge-YYYYMMDD",
    ),
    (
        "lab-alert",
        "Agent 任务失败率连续五分钟超过 8% 时触发告警。",
        "失败率超过多少会触发告警？",
        "8%",
        "任务失败率 8% 告警",
    ),
    (
        "lab-latency",
        "短回答服务的 p95 延迟目标为 6 秒以内。",
        "短回答服务的 p95 延迟目标是多少？",
        "6",
        "短回答 p95 6秒",
    ),
    (
        "lab-seed",
        "正式可复现实验默认随机种子为 2026。",
        "正式实验默认随机种子是多少？",
        "2026",
        "可复现实验 随机种子 2026",
    ),
    (
        "lab-budget",
        "本季度本地模型实验磁盘预算上限为 40GB。",
        "本季度模型实验磁盘上限是多少？",
        "40GB",
        "模型实验 磁盘 40GB",
    ),
    (
        "lab-adapter",
        "通用对话与工具调用分别保存独立 LoRA Adapter。",
        "通用对话和工具调用是否共用同一个 Adapter？",
        "独立",
        "通用对话 工具调用 独立 Adapter",
    ),
    (
        "lab-ci",
        "持续集成只运行五条离线 smoke 预测回放，不下载模型权重。",
        "持续集成会下载模型权重吗？",
        "不",
        "持续集成 不下载 模型权重",
    ),
    (
        "lab-private",
        "用户上传的私有文档默认不进入训练数据。",
        "私有文档默认会进入训练吗？",
        "不",
        "私有文档 不进入 训练",
    ),
    (
        "lab-delete",
        "用户删除知识库后，索引与原始文件需要在 24 小时内一并清除。",
        "删除知识库后多久内清除文件？",
        "24",
        "删除知识库 24小时 清除",
    ),
    (
        "lab-format",
        "第一阶段知识库支持 TXT、Markdown 和 Python 源文件。",
        "第一阶段支持哪三类知识文件？",
        "Markdown",
        "知识库 TXT Markdown Python",
    ),
    (
        "lab-docx",
        "Word 报告生成后必须重新读取并完成页面渲染检查。",
        "Word 报告生成后的额外检查是什么？",
        "渲染",
        "Word 报告 重新读取 渲染",
    ),
    (
        "lab-pdf",
        "PDF 验收要求页面不存在文字截断、重叠或乱码。",
        "PDF 页面验收不允许出现哪类问题？",
        "乱码",
        "PDF 截断 重叠 乱码",
    ),
    (
        "lab-release",
        "候选版本只有通过冻结测试后才能标记为 release。",
        "候选版本何时才能标记 release？",
        "冻结测试",
        "候选版本 冻结测试 release",
    ),
]


def retrieval_spec(document_id: str, query: str) -> dict[str, Any]:
    corpus = [{"id": f"doc-{item[0]}", "text": item[1]} for item in DOCUMENT_SPECS]
    return {
        "query": query,
        "k": 5,
        "gold_doc_ids": [f"doc-{document_id}"],
        "corpus": corpus,
    }


def retrieval_cases() -> list[dict[str, Any]]:
    rows = []
    for index, (document_id, _text, question, answer, query) in enumerate(
        DOCUMENT_SPECS,
        start=1,
    ):
        rows.append(
            base_case(
                f"frt-{index:03d}",
                "retrieval",
                f"请根据澄川研究所资料回答：{question}",
                expect_tool=True,
                expected_tools=["search_docs"],
                forbidden_tools=["calculator", "current_time"],
                gold_answer=answer,
                answer_contains=[answer],
                success_mode="cite_and_answer",
                citations=[f"doc-{document_id}"],
                retrieval=retrieval_spec(document_id, query),
            )
        )
    return rows


def multi_step_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arithmetic = [
        (
            "lab-retention",
            "脱敏推理日志保留天数乘以 2 是多少？",
            "14*2",
            "28",
            "脱敏 推理日志 14天",
        ),
        ("lab-topk", "词法检索默认条数加 6 是多少？", "4+6", "10", "词法检索 Top-4"),
        (
            "lab-timeout",
            "只读检索超时秒数乘以 5 是多少？",
            "3*5",
            "15",
            "只读检索 超时 3秒",
        ),
        (
            "lab-upload",
            "单文件上限的数值减去 8 是多少？",
            "20-8",
            "12",
            "文本文件 上限 20MB",
        ),
        (
            "lab-chunk",
            "默认切分字符数除以 8 是多少？",
            "480/8",
            "60",
            "文档切分 480字符",
        ),
        (
            "lab-alert",
            "告警阈值的百分数数值加 2 是多少？",
            "8+2",
            "10",
            "失败率 8% 告警",
        ),
        (
            "lab-seed",
            "默认随机种子减去 2000 是多少？",
            "2026-2000",
            "26",
            "随机种子 2026",
        ),
        (
            "lab-delete",
            "清除时限小时数除以 3 是多少？",
            "24/3",
            "8",
            "删除知识库 24小时",
        ),
    ]
    for index, (document_id, question, expression, answer, query) in enumerate(
        arithmetic,
        start=1,
    ):
        rows.append(
            base_case(
                f"fms-{index:03d}",
                "multi_step",
                f"先查澄川研究所资料，再用计算器回答：{question}",
                expect_tool=True,
                expected_tools=["search_docs", "calculator"],
                forbidden_tools=["current_time"],
                expected_arguments={"calculator": {"expression": expression}},
                gold_answer=answer,
                answer_contains=[answer],
                success_mode="tool_and_answer",
                citations=[f"doc-{document_id}"],
                retrieval=retrieval_spec(document_id, query),
                min_tool_calls=2,
            )
        )
    retrieval_time = [
        (
            "lab-address",
            "Asia/Shanghai",
            "查到访客地址后，再查询当地当前时间。",
            "访客入口 星湖街",
        ),
        (
            "lab-wifi",
            "Asia/Shanghai",
            "查到访客网络名后，再查询研究所当地时间。",
            "访客网络 Chengchuan-Guest",
        ),
        (
            "lab-room",
            "Asia/Shanghai",
            "查到复盘会议室后，再查询会议所在地时间。",
            "模型复盘会 C4",
        ),
        (
            "lab-owner",
            "Asia/Shanghai",
            "查到值班短号后，再查询负责人所在地时间。",
            "值班 短号 7312",
        ),
    ]
    for index, (document_id, timezone, question, query) in enumerate(
        retrieval_time,
        start=9,
    ):
        rows.append(
            base_case(
                f"fms-{index:03d}",
                "multi_step",
                question,
                expect_tool=True,
                expected_tools=["search_docs", "current_time"],
                forbidden_tools=["calculator"],
                expected_arguments={"current_time": {"timezone": timezone}},
                success_mode="tool_and_answer",
                citations=[f"doc-{document_id}"],
                retrieval=retrieval_spec(document_id, query),
                min_tool_calls=2,
            )
        )
    calc_time = [
        ("计算 18 乘 7，然后查询 UTC 当前时间。", "18*7", "126", "UTC"),
        ("计算 225 除以 9，然后查询东京时间。", "225/9", "25", "Asia/Tokyo"),
        ("计算 64 的平方根，然后查询伦敦时间。", "64**0.5", "8", "Europe/London"),
        ("计算 37 加 58，然后查询纽约时间。", "37+58", "95", "America/New_York"),
    ]
    for index, (question, expression, answer, timezone) in enumerate(
        calc_time,
        start=13,
    ):
        rows.append(
            base_case(
                f"fms-{index:03d}",
                "multi_step",
                question,
                expect_tool=True,
                expected_tools=["calculator", "current_time"],
                forbidden_tools=["search_docs"],
                expected_arguments={
                    "calculator": {"expression": expression},
                    "current_time": {"timezone": timezone},
                },
                gold_answer=answer,
                answer_contains=[answer],
                success_mode="tool_and_answer",
                min_tool_calls=2,
            )
        )
    recovery_specs: list[tuple[str, str, str | None, str, str | None]] = [
        (
            "计算器第一次暂时失败时，重试计算 41 乘 6。",
            "calculator",
            "41*6",
            "246",
            None,
        ),
        (
            "计算器第一次暂时失败时，重试计算 512 除以 16。",
            "calculator",
            "512/16",
            "32",
            None,
        ),
        (
            "检索第一次暂时失败时，重试查询澄川研究所的日志保留天数。",
            "search_docs",
            None,
            "14",
            "lab-retention",
        ),
        (
            "检索第一次暂时失败时，重试查询冻结测试的发布要求。",
            "search_docs",
            None,
            "双人",
            "lab-review",
        ),
    ]
    for index, (
        question,
        tool,
        recovery_expression,
        answer,
        recovery_document_id,
    ) in enumerate(
        recovery_specs,
        start=17,
    ):
        expected_arguments = None
        if recovery_expression is not None:
            expected_arguments = {"calculator": {"expression": recovery_expression}}
        retrieval = (
            retrieval_spec(recovery_document_id, f"澄川研究所 {answer}")
            if recovery_document_id
            else None
        )
        rows.append(
            base_case(
                f"fms-{index:03d}",
                "multi_step",
                question,
                expect_tool=True,
                expected_tools=[tool],
                forbidden_tools=[
                    name
                    for name in ["calculator", "current_time", "search_docs"]
                    if name != tool
                ],
                expected_arguments=expected_arguments,
                gold_answer=answer,
                answer_contains=[answer],
                success_mode="recovery",
                citations=(
                    [f"doc-{recovery_document_id}"] if recovery_document_id else []
                ),
                retrieval=retrieval,
                expect_recovery=True,
                min_tool_calls=2,
            )
        )
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = {
        "direct_answer.jsonl": direct_cases(),
        "tool_selection.jsonl": tool_selection_cases(),
        "tool_arguments.jsonl": tool_argument_cases(),
        "retrieval.jsonl": retrieval_cases(),
        "multi_step.jsonl": multi_step_cases(),
    }
    for name, rows in groups.items():
        write_jsonl(name, rows)
    expected_counts = {
        "direct_answer": 30,
        "tool_selection": 45,
        "tool_arguments": 25,
        "retrieval": 30,
        "multi_step": 20,
    }
    manifest = build_manifest_payload(
        OUTPUT,
        split="frozen_test",
        version="agent-frozen-test-v1-candidate",
        purpose="Release-only generalization test; never use for prompt or SFT tuning.",
        frozen=True,
        expected_counts=expected_counts,
        review_status="pending_owner_signoff",
        provenance=(
            "Deterministic AI-assisted candidate authored for P1; machine validation "
            "does not replace project-owner human signoff."
        ),
    )
    path = write_manifest(OUTPUT, manifest)
    print(f"wrote {sum(len(rows) for rows in groups.values())} cases and {path}")


if __name__ == "__main__":
    main()
