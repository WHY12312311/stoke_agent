"""定义股票多智能体系统的状态结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

# 单个 thread 最多保留的消息条数，超出后自动丢弃最旧的消息
_MAX_MESSAGES = 50


def _bounded_add_messages(
    existing: Sequence[AnyMessage], new: Sequence[AnyMessage]
) -> list[AnyMessage]:
    """带上限的消息 reducer：先执行 add_messages 追加/更新，再裁剪到 _MAX_MESSAGES 条。

    超出上限时保留最新的 _MAX_MESSAGES 条，从根源上控制 checkpoint 内存增长。
    注意：add_messages 要求入参为 list，需显式转换，否则传入其他 Sequence 子类时会报错。
    """
    # add_messages 返回值类型为 Messages（可能是单条或列表），用 list() 强制转换为 list[AnyMessage]
    raw = add_messages(list(existing), list(new))
    merged: list[AnyMessage] = list(raw) if isinstance(raw, (list, tuple)) else [raw]  # type: ignore[arg-type]
    if len(merged) > _MAX_MESSAGES:
        merged = merged[-_MAX_MESSAGES:]
    return merged


@dataclass
class InputState:
    """外部输入状态：用户发送的消息列表。"""

    messages: Annotated[Sequence[AnyMessage], _bounded_add_messages] = field(
        default_factory=list
    )
    """
    对话消息列表，使用 add_messages 注解保证追加语义。
    典型流程：
    1. HumanMessage  - 用户输入（如"买了100股茅台"或"分析我的持仓"）
    2. AIMessage     - Supervisor 分配任务
    3. AIMessage     - 各子 Agent 分析结果
    4. AIMessage     - Supervisor 汇总最终结论
    """


@dataclass
class StockAnalysisState(InputState):
    """完整的股票分析状态，扩展 InputState，存储各子 Agent 的分析结果。"""

    is_last_step: IsLastStep = field(default=False)
    """是否已到达最大递归步数，由 LangGraph 框架管理。"""

    # ---- 任务模式 ----

    task_mode: Optional[str] = field(default=None)
    """
    任务模式：
    - 'record_trade'       : 记录买卖交易
    - 'analyze_portfolio'  : 分析当前持仓
    - 'recommend_stocks'   : 根据预算推荐选股
    """

    # ---- 交易记录相关 ----

    trade_parse_result: Optional[str] = field(default=None)
    """从口语化描述中解析出的交易信息（JSON 字符串），由 trade_recorder 节点填充。"""

    trade_result_message: Optional[str] = field(default=None)
    """交易记录操作结果的文字描述。"""

    # ---- 持仓分析相关 ----

    portfolio_summary: Optional[str] = field(default=None)
    """当前持仓摘要，由 supervisor_entry 节点从持仓文件读取后填充。"""

    symbols_to_analyze: list[str] = field(default_factory=list)
    """需要分析的股票代码列表（从持仓中提取）。"""

    # ---- 各子 Agent 的分析结果（按股票代码存储）----

    market_data: Optional[str] = field(default=None)
    """行情监控 Agent 的输出：所有持仓股票的实时行情摘要。"""

    technical_analysis: Optional[str] = field(default=None)
    """技术分析 Agent 的输出：MA、RSI、MACD 等技术指标信号。"""

    news_sentiment: Optional[str] = field(default=None)
    """新闻情绪 Agent 的输出：近期新闻情绪评分与关键事件摘要。"""

    fundamental_analysis: Optional[str] = field(default=None)
    """基本面分析 Agent 的输出：财务指标、行业地位、估值分析。"""

    final_conclusion: Optional[str] = field(default=None)
    """Supervisor 汇总后的最终投资建议（含持仓维度的买入/持有/卖出建议）。"""

    # ---- 选股推荐相关 ----

    budget: Optional[float] = field(default=None)
    """用户给出的投资预算（元），由 supervisor_entry 从用户消息中解析后填充。"""

    recommend_result: Optional[str] = field(default=None)
    """选股推荐 Agent 的输出：推荐的 5 支股票及买入数量、推荐理由。"""

    # ---- 问答分析相关 ----

    qa_result: Optional[str] = field(default=None)
    """QA Agent（ReAct 架构）的输出：针对用户问题的分析解答。"""
