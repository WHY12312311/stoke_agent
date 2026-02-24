"""股票多智能体系统 - Supervisor 架构。

架构说明：
- 系统支持四种模式：
  1. 交易记录模式（record_trade）：解析口语化买卖描述、入金/出金，更新持仓及账户余额
  2. 持仓分析模式（analyze_portfolio）：对所有持仓股票进行全面分析
  3. 选股推荐模式（recommend_stocks）：根据预算推荐股票（支持显式预算或自动使用账户余额）
  4. 问答分析模式（qa）：ReAct 架构，按需调用工具回答问题；追问时 LLM 直接基于历史回答，无需工具调用

资金管理：
- Portfolio 维护 cash_balance（账户现金余额）
- 买入时自动扣减现金，卖出时自动回收现金
- 支持入金（deposit）/ 出金（withdraw）操作
- 推荐股票时：优先使用用户指定预算，无显式预算时回退到账户余额
- 支持"再投X万"追加投资语义

智能交易：
- 支持"全部卖出"、"清仓"、"卖一半"等语义化卖出
- LLM 在解析交易时会被注入当前持仓明细（含各股票具体股数）

数据流：
  __start__
      ↓
  supervisor_entry（识别意图，加载持仓）
      ↓
  [record_trade]   [recommend_stocks]   [qa]          [analyze_portfolio]
  trade_recorder   stock_recommender    qa_agent       ↓（并行）
      ↓                  ↓                ↓     [market_monitor, technical_analysis,
  __end__            __end__           __end__   news_sentiment, fundamental_analysis]
                                                        ↓
                                               supervisor_conclude（汇总报告）
                                                        ↓
                                                    __end__
"""

import json
import logging
import re
from typing import Any, Dict, Literal

from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import StateGraph
from langgraph.types import Send
from langgraph.runtime import Runtime

from common.context import Context
from common.utils import load_chat_model
from stoke_multi_agent.agents import (
    fundamental_analysis_node,
    market_monitor_node,
    news_sentiment_node,
    qa_agent_node,
    stock_recommender_node,
    technical_analysis_node,
    trade_recorder_node,
)
# follow_up_handler_node 已合并到 qa_agent_node，不再单独导入
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
- "record_trade"：用户在描述买入或卖出股票的操作（如"买了100股茅台"、"卖掉了我的腾讯"），或者进行资金操作（如"入金5万"、"充值3万"、"出金1万"、"提现"）
- "analyze_portfolio"：用户想要对当前持仓进行全面分析（如"分析我的持仓"、"帮我全面看看我的股票"）
- "recommend_stocks"：用户想让你推荐要买哪些股票（如"我有5万元，帮我推荐几支股票"、"预算3万，买什么好"、"帮我推荐几支股票"、"给我选几个股"）
- "qa"：其他所有情况，包括：针对某只股票提问/查询/分析（如"茅台现在多少钱"、"比亚迪基本面怎么样"）、追问上一轮回答（如"帮我详细说说第二条"、"为什么这么建议"、"那茅台呢"）、泛问题（如"A股最近行情如何"）

对话历史：
{history_text}
最新用户消息：{user_message}

只返回 "record_trade"、"analyze_portfolio"、"recommend_stocks" 或 "qa"，不要有其他内容。"""

    response = await model.ainvoke([HumanMessage(content=intent_prompt)])
    raw_intent = response.content if isinstance(response.content, str) else str(response.content)
    task_mode = raw_intent.strip().strip('"').strip("'")

    # 容错处理
    if task_mode not in ("record_trade", "analyze_portfolio", "recommend_stocks", "qa"):
        # 简单关键词兜底
        buy_sell_keywords = ["买", "卖", "购入", "清仓", "加仓", "减仓", "买入", "卖出",
                             "入金", "充值", "转入", "出金", "提现", "取出"]
        recommend_keywords = ["推荐", "预算", "帮我选", "买什么", "选股", "配置"]
        if any(kw in user_message for kw in recommend_keywords):
            task_mode = "recommend_stocks"
        elif any(kw in user_message for kw in buy_sell_keywords):
            task_mode = "record_trade"
        else:
            # 其余所有情况（包括追问、单股查询、泛问题）统一走 qa
            task_mode = "qa"

    logger.info(f"[Supervisor] Task mode: {task_mode}")

    # 加载持仓数据
    portfolio = await async_load_portfolio()
    portfolio_summary = get_portfolio_summary(portfolio)
    symbols_to_analyze = list(portfolio.positions.keys())

    # 解析预算金额（仅 recommend_stocks 模式需要）
    budget: float = 0.0
    if task_mode == "recommend_stocks":
        # First try to extract explicit budget from user message
        budget = _parse_budget(user_message)

        # If no explicit budget, fall back to account cash balance
        if budget <= 0 and portfolio.cash_balance > 0:
            budget = portfolio.cash_balance
            logger.info(f"[Supervisor] No explicit budget, using account cash balance: {budget} CNY")
        elif budget > 0:
            # User gave an explicit budget - check if it's additional investment
            # Keywords indicating additional investment on top of existing balance
            add_keywords = ["再投", "追加", "再加", "额外", "另外", "再投入", "多投"]
            if any(kw in user_message for kw in add_keywords):
                budget = budget + portfolio.cash_balance
                logger.info(f"[Supervisor] Additional investment detected, total budget: {budget} CNY")

        logger.info(f"[Supervisor] Final budget: {budget} CNY")

    # 生成提示消息
    if task_mode == "qa":
        dispatch_msg = AIMessage(content="🔍 正在为您查询分析，请稍候...")
    elif task_mode == "record_trade":
        dispatch_msg = AIMessage(content="📝 正在解析您的操作记录...")
    elif task_mode == "recommend_stocks":
        if budget > 0:
            # Show budget source info
            budget_source = ""
            if portfolio.cash_balance > 0 and _parse_budget(user_message) <= 0:
                budget_source = f"（使用账户现金余额）"
            dispatch_msg = AIMessage(
                content=f"🔎 正在为您筛选推荐股票...\n"
                        f"投资预算：{budget:,.0f} 元{budget_source}\n"
                        f"分析维度：行情筛选 | 基本面评估 | 新闻情绪 | 估值分析"
            )
        else:
            dispatch_msg = AIMessage(
                content="⚠️ 未能识别到预算金额，且账户无可用现金余额。\n"
                        "请重新描述，例如：'我有 5 万元，帮我推荐 5 支股票'\n"
                        "或者先入金：'入金 5 万'"
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
    - qa 模式：路由到 qa_agent（ReAct 架构，按需调用工具；追问时 LLM 直接回答不调用工具）
    - analyze_portfolio 模式（有持仓）：并行路由到4个分析节点
    - analyze_portfolio 模式（无持仓）：路由到 no_portfolio

    Returns:
        节点名称字符串或 Send 对象列表（实现并行）。
    """
    if state.task_mode == "record_trade":
        return ["trade_recorder"]
    elif state.task_mode == "recommend_stocks":
        # 预算为 None 或 0 时也路由到 stock_recommender，由节点内部处理错误
        return ["stock_recommender"]
    elif state.task_mode == "qa":
        # QA Agent：ReAct 架构，按需调用工具回答用户问题
        return ["qa_agent"]
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
builder.add_node("no_portfolio", no_portfolio_node)
builder.add_node("stock_recommender", stock_recommender_node)
builder.add_node("qa_agent", qa_agent_node)
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
        "trade_recorder",
        "no_portfolio",
        "stock_recommender",
        "qa_agent",
        "market_monitor",
        "technical_analysis",
        "news_sentiment",
        "fundamental_analysis",
    ],
)

# 交易记录完成 -> 结束
builder.add_edge("trade_recorder", "__end__")

# 无持仓提示 -> 结束
builder.add_edge("no_portfolio", "__end__")

# 选股推荐完成 -> 结束
builder.add_edge("stock_recommender", "__end__")

# QA Agent（ReAct 问答）完成 -> 结束
builder.add_edge("qa_agent", "__end__")

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
