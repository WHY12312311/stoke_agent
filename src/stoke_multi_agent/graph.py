"""股票多智能体系统 - Supervisor 架构。

架构说明：
- 系统支持两种模式：
  1. 交易记录模式（record_trade）：解析口语化买卖描述，更新持仓
  2. 持仓分析模式（analyze_portfolio）：对所有持仓股票进行全面分析

数据流：
  __start__
      ↓
  supervisor_entry（识别意图，加载持仓）
      ↓
  [record_trade 模式]          [analyze_portfolio 模式]
  trade_recorder               ↓（并行）
      ↓                [market_monitor, technical_analysis,
  __end__               news_sentiment, fundamental_analysis]
                               ↓
                        supervisor_conclude（汇总报告）
                               ↓
                           __end__
"""

import json
import logging
import re
from typing import Any, Dict, Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langgraph.graph import StateGraph
from langgraph.types import Send
from langgraph.runtime import Runtime

from common.context import Context
from common.utils import load_chat_model
from stoke_multi_agent.agents import (
    fundamental_analysis_node,
    market_monitor_node,
    news_sentiment_node,
    stock_recommender_node,
    technical_analysis_node,
    trade_recorder_node,
)
from stoke_multi_agent.portfolio import get_portfolio_summary, async_load_portfolio, load_portfolio
from stoke_multi_agent.state import InputState, StockAnalysisState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supervisor Entry Node：识别用户意图，路由到对应模式
# ---------------------------------------------------------------------------

async def supervisor_entry(
    state: StockAnalysisState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Supervisor 入口节点：识别用户意图（记录交易 or 分析持仓），加载持仓数据。

    Args:
        state: 当前状态，包含用户消息。
        runtime: LangGraph 运行时。

    Returns:
        更新 task_mode、portfolio_summary、symbols_to_analyze 和 messages 字段的状态 patch。
    """
    # 获取最新用户消息
    user_message = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    logger.info(f"[Supervisor] Processing: {user_message[:100]}")

    # 构建最近对话历史（最多取最近 6 条，用于意图识别上下文）
    recent_messages = state.messages[-6:] if len(state.messages) > 6 else state.messages
    history_text = ""
    for msg in recent_messages:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # 助手消息截断，避免 prompt 过长
        if role == "助手" and len(content) > 200:
            content = content[:200] + "..."
        history_text += f"{role}：{content}\n"

    # 用 LLM 识别用户意图
    model = load_chat_model(runtime.context.model)
    intent_prompt = f"""请根据对话历史和最新用户消息，判断用户的意图，只返回以下四个值之一：
- "record_trade"：用户在描述买入或卖出股票的操作（如"买了100股茅台"、"卖掉了我的腾讯"）
- "analyze_portfolio"：用户想要分析持仓、查看行情、获取投资建议（如"分析我的持仓"、"帮我看看现在该怎么操作"）
- "recommend_stocks"：用户给出了一笔预算，想让你推荐要买哪些股票（如"我有5万元，帮我推荐几支股票"、"预算3万，买什么好"）
- "follow_up"：用户是在追问、追加说明或针对上一轮回答提问（如"帮我详细说说第二条"、"为什么这么建议"、"那茅台呢"、"能解释一下吗"）

对话历史：
{history_text}
最新用户消息：{user_message}

只返回 "record_trade"、"analyze_portfolio"、"recommend_stocks" 或 "follow_up"，不要有其他内容。"""

    response = await model.ainvoke([HumanMessage(content=intent_prompt)])
    raw_intent = response.content if isinstance(response.content, str) else str(response.content)
    task_mode = raw_intent.strip().strip('"').strip("'")

    # 容错处理
    if task_mode not in ("record_trade", "analyze_portfolio", "recommend_stocks", "follow_up"):
        # 简单关键词兜底
        buy_sell_keywords = ["买", "卖", "购入", "清仓", "加仓", "减仓", "买入", "卖出"]
        recommend_keywords = ["推荐", "预算", "帮我选", "买什么", "选股", "配置"]
        follow_up_keywords = ["为什么", "详细", "解释", "说说", "那", "呢", "能不能", "怎么理解"]
        if any(kw in user_message for kw in recommend_keywords):
            task_mode = "recommend_stocks"
        elif any(kw in user_message for kw in buy_sell_keywords):
            task_mode = "record_trade"
        elif len(state.messages) > 2 and any(kw in user_message for kw in follow_up_keywords):
            # 有历史对话且包含追问关键词，判定为追问
            task_mode = "follow_up"
        else:
            task_mode = "analyze_portfolio"

    logger.info(f"[Supervisor] Task mode: {task_mode}")

    # 加载持仓数据
    portfolio = await async_load_portfolio()
    portfolio_summary = get_portfolio_summary(portfolio)
    symbols_to_analyze = list(portfolio.positions.keys())

    # 解析预算金额（仅 recommend_stocks 模式需要）
    budget: float = 0.0
    if task_mode == "recommend_stocks":
        # 用正则从用户消息中提取数字金额（支持 "5万"、"50000"、"5w" 等写法）
        budget = _parse_budget(user_message)
        logger.info(f"[Supervisor] Parsed budget: {budget} CNY")

    # 生成提示消息
    if task_mode == "follow_up":
        dispatch_msg = AIMessage(content="💬 正在为您解答...")
    elif task_mode == "record_trade":
        dispatch_msg = AIMessage(content="📝 正在解析您的交易记录...")
    elif task_mode == "recommend_stocks":
        if budget > 0:
            dispatch_msg = AIMessage(
                content=f"� 正在为您筛选推荐股票...\n"
                        f"投资预算：{budget:,.0f} 元\n"
                        f"分析维度：行情筛选 | 基本面评估 | 新闻情绪 | 估值分析"
            )
        else:
            dispatch_msg = AIMessage(
                content="⚠️ 未能识别到预算金额，请重新描述，例如：'我有 5 万元，帮我推荐 5 支股票'"
            )
    else:
        # analyze_portfolio 模式
        if symbols_to_analyze:
            dispatch_msg = AIMessage(
                content=f"� 正在对您的持仓进行全面分析...\n"
                        f"持仓股票：{', '.join(symbols_to_analyze)}\n"
                        f"分析维度：行情监控 | 技术分析 | 新闻情绪 | 基本面分析"
            )
        else:
            dispatch_msg = AIMessage(content="⚠️ 当前无持仓记录，请先告诉我您买了哪些股票。")
    return {
        "task_mode": task_mode,
        "budget": budget,
        "portfolio_summary": portfolio_summary,
        "symbols_to_analyze": symbols_to_analyze,
        "messages": [dispatch_msg],
    }

# ---------------------------------------------------------------------------
# Budget Parser：从口语化文本中提取预算金额
# ---------------------------------------------------------------------------

def _parse_budget(text: str) -> float:
    """从口语化文本中提取预算金额（元）。

    支持格式：
    - "5万"、"5w"、"5W" -> 50000
    - "50000"、"50,000" -> 50000
    - "1.5万" -> 15000
    - "100k" -> 100000
    """
    text = text.replace(",", "").replace("，", "")
    # 匹配 "数字+万/w/W" 格式
    m = re.search(r"([\d.]+)\s*[万wW]", text)
    if m:
        return float(m.group(1)) * 10000
    # 匹配 "数字+k/K" 格式
    m = re.search(r"([\d.]+)\s*[kK]", text)
    if m:
        return float(m.group(1)) * 1000
    # 匹配纯数字（>= 1000 才认为是预算）
    m = re.search(r"([\d.]+)", text)
    if m:
        val = float(m.group(1))
        if val >= 1000:
            return val
    return 0.0


# ---------------------------------------------------------------------------
# Supervisor Route：根据 task_mode 决定下一步
# ---------------------------------------------------------------------------

def supervisor_route(
    state: StockAnalysisState,
) -> list:
    """根据任务模式路由到对应节点。

    - record_trade 模式：路由到 trade_recorder
    - recommend_stocks 模式：路由到 stock_recommender
    - analyze_portfolio 模式（有持仓）：并行路由到4个分析节点
    - analyze_portfolio 模式（无持仓）：路由到 no_portfolio

    Returns:
        节点名称字符串或 Send 对象列表（实现并行）。
    """
    if state.task_mode == "follow_up":
        return ["follow_up_handler"]
    elif state.task_mode == "record_trade":
        return ["trade_recorder"]
    elif state.task_mode == "recommend_stocks":
        # 预算为 None 或 0 时也路由到 stock_recommender，由节点内部处理错误
        return ["stock_recommender"]
    elif state.symbols_to_analyze:
        # 并行触发4个分析节点，共享同一状态
        return [
            Send("market_monitor", state),
            Send("technical_analysis", state),
            Send("news_sentiment", state),
            Send("fundamental_analysis", state),
        ]
    else:
        return ["no_portfolio"]


# ---------------------------------------------------------------------------
# Follow-up Handler Node：处理追问，基于历史对话直接回答
# ---------------------------------------------------------------------------

async def follow_up_handler_node(
    state: StockAnalysisState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """追问处理节点：基于完整对话历史，直接用 LLM 回答用户的追问。

    不触发任何数据拉取或分析，直接利用 messages 中的历史上下文回答。
    """
    model = load_chat_model(runtime.context.model)

    # 将历史消息直接传给 LLM（LangChain 消息格式兼容）
    system_prompt = HumanMessage(
        content="你是一位专业的 A 股投资顾问。请根据对话历史，回答用户的追问。"
                "回答要简洁、专业，如果涉及具体数据请说明数据来源时间。"
    )
    # 只取最近 20 条消息传给 LLM，避免历史过长导致 token 超限
    max_history = 20
    recent_messages = list(state.messages[-max_history:]) if len(state.messages) > max_history else list(state.messages)
    messages_to_send: Sequence[BaseMessage] = [system_prompt] + recent_messages

    response = await model.ainvoke(messages_to_send)
    answer = response.content if isinstance(response.content, str) else str(response.content)

    return {"messages": [AIMessage(content=answer)]}


# ---------------------------------------------------------------------------
# No Portfolio Node：无持仓时的提示节点
# ---------------------------------------------------------------------------

async def no_portfolio_node(
    state: StockAnalysisState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """无持仓提示节点：当用户没有持仓时给出引导。"""
    msg = AIMessage(
        content="📭 您当前没有任何持仓记录。\n\n"
                "请先告诉我您买了哪些股票，例如：\n"
                "- 我买了100股贵州茅台，均价1800元\n"
                "- 买入50股腾讯，价格380港元\n\n"
                "记录持仓后，我就可以为您进行全面的持仓分析了。"
    )
    return {"messages": [msg]}


# ---------------------------------------------------------------------------
# Supervisor Conclude Node：汇总所有子 Agent 结果，生成最终报告
# ---------------------------------------------------------------------------

async def supervisor_conclude(
    state: StockAnalysisState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Supervisor 汇总节点：整合四个子 Agent 的分析结果，生成持仓维度的投资建议报告。

    Args:
        state: 包含所有子 Agent 分析结果的状态。
        runtime: LangGraph 运行时。

    Returns:
        更新 final_conclusion 和 messages 字段的状态 patch。
    """
    logger.info(f"[Supervisor] Generating final report for portfolio: {state.symbols_to_analyze}")

    model = load_chat_model(runtime.context.model)

    conclude_prompt = f"""你是一位资深投资组合分析师。请根据以下对各持仓股票的多维度分析，给出综合的持仓管理建议。

## 当前持仓情况
{state.portfolio_summary or "暂无持仓信息"}

## 行情监控分析
{state.market_data or "暂无数据"}

## 技术指标分析
{state.technical_analysis or "暂无数据"}

## 新闻情绪分析
{state.news_sentiment or "暂无数据"}

## 基本面分析
{state.fundamental_analysis or "暂无数据"}

---

请按以下结构输出完整的持仓分析报告：

### 一、持仓整体评估
（简要描述整体持仓的风险收益状况，2-3句话）

### 二、各股票操作建议
（对每只持仓股票分别给出：建议操作 + 核心理由 + 目标价/止损位）
格式：
**[股票名称（代码）]**
- 建议操作：买入加仓 / 持有观望 / 减仓 / 清仓
- 核心理由：（结合行情、技术、新闻、基本面综合说明，3-5条）
- 目标价：xxx 元 | 止损位：xxx 元

### 三、持仓结构优化建议
（从仓位分配、行业分散、风险对冲角度给出建议）

### 四、近期重点关注事项
（列出 3-5 个需要重点跟踪的风险点或催化剂）

### 五、风险提示
（重要免责声明和风险提示）

请确保建议具体、可操作，并充分结合持仓成本进行分析。"""

    response = await model.ainvoke([HumanMessage(content=conclude_prompt)])
    conclusion = response.content if isinstance(response.content, str) else str(response.content)

    # 构建最终报告消息
    symbols_str = "、".join(state.symbols_to_analyze) if state.symbols_to_analyze else "无"
    final_message = AIMessage(
        content=f"## 📊 持仓分析报告\n**分析股票：{symbols_str}**\n\n{conclusion}"
    )

    logger.info("[Supervisor] Final report generated")
    return {
        "final_conclusion": conclusion,
        "messages": [final_message],
    }


# ---------------------------------------------------------------------------
# Build the Supervisor Graph
# ---------------------------------------------------------------------------

builder = StateGraph(StockAnalysisState, input_schema=InputState, context_schema=Context)

# ---- 添加节点 ----
builder.add_node("supervisor_entry", supervisor_entry)
builder.add_node("trade_recorder", trade_recorder_node)
builder.add_node("follow_up_handler", follow_up_handler_node)
builder.add_node("no_portfolio", no_portfolio_node)
builder.add_node("stock_recommender", stock_recommender_node)
builder.add_node("market_monitor", market_monitor_node)
builder.add_node("technical_analysis", technical_analysis_node)
builder.add_node("news_sentiment", news_sentiment_node)
builder.add_node("fundamental_analysis", fundamental_analysis_node)
builder.add_node("supervisor_conclude", supervisor_conclude)

# ---- 定义边 ----

# 入口 -> Supervisor 识别意图
builder.add_edge("__start__", "supervisor_entry")

# Supervisor 根据意图路由（支持并行 Send）
builder.add_conditional_edges(
    "supervisor_entry",
    supervisor_route,
    [
        "follow_up_handler",
        "trade_recorder",
        "no_portfolio",
        "stock_recommender",
        "market_monitor",
        "technical_analysis",
        "news_sentiment",
        "fundamental_analysis",
    ],
)

# 追问处理完成 -> 结束
builder.add_edge("follow_up_handler", "__end__")

# 交易记录完成 -> 结束
builder.add_edge("trade_recorder", "__end__")

# 无持仓提示 -> 结束
builder.add_edge("no_portfolio", "__end__")

# 选股推荐完成 -> 结束
builder.add_edge("stock_recommender", "__end__")

# 四个分析节点全部完成后 -> Supervisor 汇总
# LangGraph 会等待所有并行节点完成后再执行 supervisor_conclude
builder.add_edge("market_monitor", "supervisor_conclude")
builder.add_edge("technical_analysis", "supervisor_conclude")
builder.add_edge("news_sentiment", "supervisor_conclude")
builder.add_edge("fundamental_analysis", "supervisor_conclude")

# 汇总 -> 结束
builder.add_edge("supervisor_conclude", "__end__")

# 编译图（LangGraph API 平台自动处理持久化，无需自定义 checkpointer）
graph = builder.compile(name="Stock Portfolio Supervisor")
