"""定义多智能体系统的各子 Agent 节点。

子 Agent：
- trade_recorder_node      : 解析口语化买卖描述，更新持仓
- market_monitor_node      : 获取所有持仓股票的实时行情
- technical_analysis_node  : 计算并解读技术指标
- news_sentiment_node      : 搜索新闻并评估市场情绪
- fundamental_analysis_node: 获取财务指标、估值、行业信息
"""

import json
import logging
import asyncio
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from common.context import Context
from common.utils import load_chat_model
from stoke_multi_agent.portfolio import (
    apply_trade,
    async_load_portfolio,
    async_save_portfolio,
    deposit_cash,
    get_portfolio_summary,
    load_portfolio,
    Portfolio,
    withdraw_cash,
)
from stoke_multi_agent.state import StockAnalysisState
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from stoke_multi_agent.tools import (
    STOKE_BASE_TOOLS,
    _ak_call,
    _get_fundamental_data,
    _get_stock_history,
    _get_stock_price,
    _search_financial_news,
    _search_stock_by_name,
    _search_stock_news,
    compute_technical_indicators,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-Agent 0: Trade Recorder（交易记录节点）
# ---------------------------------------------------------------------------

async def trade_recorder_node(
    state: StockAnalysisState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """交易记录节点：从口语化描述中解析买卖信息，更新持仓。

    使用 LLM 理解用户的口语化描述（如"买了100股茅台，均价1800"），
    提取结构化交易信息，然后更新本地持仓文件。

    Args:
        state: 当前状态，包含用户消息。
        runtime: LangGraph 运行时。

    Returns:
        更新 trade_result_message 和 messages 字段的状态 patch。
    """
    # 获取最新用户消息
    user_message = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    logger.info(f"[TradeRecorder] Parsing trade from: {user_message[:100]}")

    model = load_chat_model(runtime.context.model)

    # ---------------------------------------------------------------
    # Step 1：LLM 只负责提取「交易意图」和「公司名称/关键词」
    # 不让 LLM 猜股票代码，避免幻觉导致买错股票
    # ---------------------------------------------------------------
    # Load portfolio upfront so LLM can see current holdings for smart sell
    portfolio = await async_load_portfolio()
    portfolio_info = get_portfolio_summary(portfolio)

    # Build position details for the LLM (so it knows exact share counts)
    positions_detail = ""
    if portfolio.positions:
        pos_lines = []
        for sym, pos in portfolio.positions.items():
            pos_lines.append(f"  - {pos.company_name}（{sym}）：持有 {pos.shares:.0f} 股，均价 {pos.avg_cost:.2f} 元")
        positions_detail = "\n".join(pos_lines)
    else:
        positions_detail = "  当前无持仓"

    parse_prompt = f"""请从用户的口语化描述中提取所有操作信息，返回 JSON 数组格式。

用户描述：{user_message}

当前账户信息：
- 现金余额：{portfolio.cash_balance:.2f} 元
- 持仓明细：
{positions_detail}

请返回如下 JSON 数组（每个操作一个对象，如果某字段无法确定，填 null）：
[
  {{
    "action": "buy" 或 "sell" 或 "deposit"（入金） 或 "withdraw"（出金）,
    "company_query": "用户提到的公司名称或简称（买卖时必填，入金/出金时填 null）",
    "shares": 股数（数字，A股1手=100股，入金/出金时填 null）,
    "price": 成交价格（元/股，如果用户没说价格填 null，入金/出金时填 null）,
    "amount": 金额（仅 deposit/withdraw 时填写，单位：元）
  }}
]

重要规则：
- 如果用户说"全部卖出"、"清仓"、"都卖了"某只股票，shares 必须填该股票的实际持仓股数（参考上面的持仓明细）
- 如果用户说"卖一半"，shares 填持仓股数的一半（取整到100的整数倍）
- 如果用户说"入金5万"、"充值3万"、"转入1万"，action 填 "deposit"，amount 填对应金额（如5万=50000）
- 如果用户说"出金1万"、"提现"、"取出"，action 填 "withdraw"，amount 填对应金额
- company_query 只填用户原文中的公司名称，**不要填股票代码**
- 如果用户说"1手"，shares = 100
- 只返回 JSON 数组，不要有其他内容
"""
    response = await model.ainvoke([HumanMessage(content=parse_prompt)])
    raw = response.content if isinstance(response.content, str) else str(response.content)

    # 清理 LLM 返回的 JSON（去掉 markdown 代码块）
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        trade_list = json.loads(raw)
        # 兼容 LLM 返回单个对象的情况
        if isinstance(trade_list, dict):
            trade_list = [trade_list]
        if not isinstance(trade_list, list) or len(trade_list) == 0:
            raise ValueError(f"Expected a non-empty JSON array, got: {type(trade_list)}")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[TradeRecorder] JSON parse failed: {e}, raw: {raw}")
        error_msg = "❌ 无法解析交易信息，请重新描述，例如：'买了100股贵州茅台，均价1800元'"
        return {
            "trade_result_message": error_msg,
            "messages": [AIMessage(content=error_msg)],
        }

    # ---------------------------------------------------------------
    # Step 2 & 3：逐笔处理每条操作
    # ---------------------------------------------------------------
    result_messages = []

    for trade_info in trade_list:
        action = trade_info.get("action")

        # --- Handle deposit / withdraw ---
        if action == "deposit":
            amount = float(trade_info.get("amount") or 0)
            result_msg = deposit_cash(portfolio, amount)
            logger.info(f"[TradeRecorder] Deposit: {result_msg}")
            result_messages.append(result_msg)
            continue
        elif action == "withdraw":
            amount = float(trade_info.get("amount") or 0)
            result_msg = withdraw_cash(portfolio, amount)
            logger.info(f"[TradeRecorder] Withdraw: {result_msg}")
            result_messages.append(result_msg)
            continue

        # --- Handle buy / sell ---
        company_query = (trade_info.get("company_query") or "").strip()
        shares = trade_info.get("shares")
        price = trade_info.get("price")

        if not all([action, company_query, shares]):
            result_messages.append(f"❌ 交易信息不完整（{company_query or '未知'}）：请提供操作方向（买/卖）、公司名称、股数")
            continue

        # Step 2：通过工具查询真实股票代码（不依赖 LLM 猜测）
        logger.info(f"[TradeRecorder] Looking up stock code for: '{company_query}'")
        lookup_result = await _search_stock_by_name(company_query)
        matches = lookup_result.get("matches", [])

        if not matches:
            result_messages.append(
                f"❌ 未找到与 '{company_query}' 匹配的 A 股股票，"
                f"请检查公司名称是否正确，或直接提供6位股票代码"
            )
            continue

        if len(matches) > 1:
            exact = [m for m in matches if m.get("match_type") == "exact"]
            if exact:
                best_match = exact[0]
            else:
                candidates = "\n".join(
                    f"  - {m['name']}（{m['symbol']}）" for m in matches
                )
                result_messages.append(
                    f"⚠️ '{company_query}' 匹配到多支股票，请指定具体名称或代码：\n{candidates}"
                )
                continue
        else:
            best_match = matches[0]

        symbol = best_match["symbol"]
        company_name = best_match["name"]
        logger.info(f"[TradeRecorder] Resolved: '{company_query}' -> {company_name}（{symbol}）")

        # Step 3：如果没有价格，用当前市价
        if not price:
            logger.info(f"[TradeRecorder] No price provided, fetching current price for {symbol}")
            market_data = await _get_stock_price(symbol)
            price = market_data.get("price", 0)
            if price == 0:
                result_messages.append(f"❌ 无法获取 {company_name}（{symbol}）的当前价格，请手动提供成交价格")
                continue

        # 应用交易到持仓（原地修改 portfolio，最后统一保存）
        result_msg = apply_trade(
            portfolio=portfolio,
            symbol=str(symbol).upper(),
            company_name=str(company_name),
            action=str(action),
            shares=float(shares or 0),
            price=float(price or 0),
        )
        logger.info(f"[TradeRecorder] Trade applied: {result_msg}")
        result_messages.append(result_msg)

    # 统一异步保存持仓（避免每笔交易都触发同步 IO）
    try:
        await async_save_portfolio(portfolio)
        logger.info(f"[TradeRecorder] Portfolio saved successfully")
    except Exception as e:
        save_error = f"⚠️ 持仓保存失败：{type(e).__name__}: {e}"
        logger.error(f"[TradeRecorder] {save_error}")
        result_messages.append(save_error)

    # 附加当前持仓摘要
    portfolio_summary = get_portfolio_summary(portfolio)
    combined_result = "\n".join(result_messages)
    full_message = f"{combined_result}\n\n{portfolio_summary}"

    logger.info(f"[TradeRecorder] All trades processed: {len(trade_list)} trades")
    return {
        "trade_result_message": combined_result,
        "messages": [AIMessage(content=full_message)],
    }


# ---------------------------------------------------------------------------
# Sub-Agent 1: Market Monitor（行情监控节点）
# ---------------------------------------------------------------------------

async def market_monitor_node(
    state: StockAnalysisState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """行情监控节点：获取所有持仓股票的实时行情，结合持仓成本分析盈亏。

    Args:
        state: 当前状态，需要 symbols_to_analyze 和 portfolio_summary 字段。
        runtime: LangGraph 运行时。

    Returns:
        更新 market_data 字段的状态 patch。
    """
    symbols = state.symbols_to_analyze or []
    if not symbols:
        return {"market_data": "无持仓股票，跳过行情分析。"}

    logger.info(f"[MarketMonitor] Fetching market data for {symbols}")

    # 并发获取所有持仓股票的行情
    import asyncio
    tasks = [_get_stock_price(symbol) for symbol in symbols]
    market_results = await asyncio.gather(*tasks, return_exceptions=True)

    # 整理行情数据
    market_data_list = []
    for symbol, result in zip(symbols, market_results):
        if isinstance(result, Exception):
            error_msg = f"{type(result).__name__}: {result}"
            logger.warning(f"[MarketMonitor] Failed to get price for {symbol}: {error_msg}")
            market_data_list.append({"symbol": symbol, "error": error_msg, "error_type": type(result).__name__})
        else:
            market_data_list.append(result)

    # 用 LLM 结合持仓成本生成行情分析
    model = load_chat_model(runtime.context.model)
    prompt = f"""你是一位专业的股票行情分析师。请根据以下持仓信息和实时行情数据，分析每只股票的当前盈亏状况。

## 持仓信息
{state.portfolio_summary or "暂无持仓信息"}

## 实时行情数据
{json.dumps(market_data_list, ensure_ascii=False, indent=2)}

⚠️ 重要提示：如果某只股票的数据中包含 "error" 字段，说明该股票的行情数据获取失败，请在分析中明确说明：
"[股票代码] 行情数据获取失败，原因：[error 字段内容]，无法进行盈亏分析。"
不要对获取失败的股票编造或推测数据。

请对每只股票分析：
1. 当前价格与持仓均价的对比（盈亏比例）
2. 今日涨跌幅及成交量情况
3. 当前市值估算

请用简洁的中文输出，每只股票一段。"""

    response = await model.ainvoke([HumanMessage(content=prompt)])
    summary = response.content if isinstance(response.content, str) else str(response.content)

    logger.info(f"[MarketMonitor] Market analysis completed for {symbols}")
    return {"market_data": summary}


# ---------------------------------------------------------------------------
# Sub-Agent 2: Technical Analysis（技术分析节点）
# ---------------------------------------------------------------------------

async def technical_analysis_node(
    state: StockAnalysisState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """技术分析节点：计算所有持仓股票的技术指标并解读信号。

    Args:
        state: 当前状态，需要 symbols_to_analyze 字段。
        runtime: LangGraph 运行时。

    Returns:
        更新 technical_analysis 字段的状态 patch。
    """
    symbols = state.symbols_to_analyze or []
    if not symbols:
        return {"technical_analysis": "无持仓股票，跳过技术分析。"}

    logger.info(f"[TechnicalAnalysis] Computing indicators for {symbols}")

    # 并发获取历史数据并计算指标
    import asyncio
    history_tasks = [_get_stock_history(symbol, period="3mo") for symbol in symbols]
    histories = await asyncio.gather(*history_tasks, return_exceptions=True)

    indicators_list = []
    for symbol, history in zip(symbols, histories):
        if isinstance(history, Exception):
            error_msg = f"{type(history).__name__}: {history}"
            logger.warning(f"[TechnicalAnalysis] Failed to get history for {symbol}: {error_msg}")
            indicators_list.append({"symbol": symbol, "error": error_msg, "error_type": type(history).__name__})
        else:
            indicators = compute_technical_indicators(history)  # type: ignore[arg-type]
            indicators_list.append(indicators)

    # 用 LLM 解读技术指标
    model = load_chat_model(runtime.context.model)
    prompt = f"""你是一位专业的技术分析师。请根据以下各股票的技术指标，分析每只股票的技术面走势。

## 技术指标数据
{json.dumps(indicators_list, ensure_ascii=False, indent=2)}

## 持仓信息（供参考）
{state.portfolio_summary or "暂无持仓信息"}

⚠️ 重要提示：如果某只股票的数据中包含 "error" 字段，说明该股票的历史行情数据获取失败，请在分析中明确说明：
"[股票代码] 技术指标数据获取失败，原因：[error 字段内容]，无法进行技术分析。"
不要对获取失败的股票编造或推测技术指标。

请对每只股票分析：
1. 趋势判断（MA5 vs MA20 金叉/死叉）
2. 动量状态（RSI 超买/超卖/中性）
3. MACD 信号（多头/空头）
4. 综合技术面评级（强势/中性/弱势）

请用简洁的中文输出，每只股票一段。"""

    response = await model.ainvoke([HumanMessage(content=prompt)])
    analysis = response.content if isinstance(response.content, str) else str(response.content)

    logger.info(f"[TechnicalAnalysis] Analysis completed for {symbols}")
    return {"technical_analysis": analysis}


# ---------------------------------------------------------------------------
# Sub-Agent 3: News Sentiment（新闻情绪节点）
# ---------------------------------------------------------------------------

async def news_sentiment_node(
    state: StockAnalysisState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """新闻情绪节点：搜索所有持仓股票的近期新闻并评估市场情绪。

    Args:
        state: 当前状态，需要 symbols_to_analyze 字段。
        runtime: LangGraph 运行时。

    Returns:
        更新 news_sentiment 字段的状态 patch。
    """
    symbols = state.symbols_to_analyze or []
    if not symbols:
        return {"news_sentiment": "无持仓股票，跳过新闻情绪分析。"}

    logger.info(f"[NewsSentiment] Searching news for {symbols}")

    # 从持仓中获取公司名称映射
    portfolio = await async_load_portfolio()

    # 并发搜索所有股票的新闻
    news_tasks = []
    for symbol in symbols:
        pos = portfolio.positions.get(symbol)
        company_name = pos.company_name if pos else None
        news_tasks.append(_search_stock_news(symbol, company_name=company_name))

    news_results = await asyncio.gather(*news_tasks, return_exceptions=True)

    news_data_list = []
    for symbol, result in zip(symbols, news_results):
        if isinstance(result, Exception):
            error_msg = f"{type(result).__name__}: {result}"
            logger.warning(f"[NewsSentiment] Failed to get news for {symbol}: {error_msg}")
            news_data_list.append({"symbol": symbol, "error": error_msg, "error_type": type(result).__name__})
        else:
            news_data_list.append(result)

    # 用 LLM 生成情绪分析
    model = load_chat_model(runtime.context.model)
    all_headlines = []
    for nd in news_data_list:
        for h in nd.get("headlines", []):
            all_headlines.append(f"[{nd.get('company_name', nd.get('symbol'))}] {h}")

    prompt = f"""你是一位专业的财经新闻分析师。请根据以下新闻标题，分析各股票的市场情绪和舆论走向。

## 近期新闻标题
{chr(10).join(f'- {h}' for h in all_headlines) or '暂无新闻数据'}

## 情绪评分数据
{json.dumps([{
    'symbol': nd.get('symbol'),
    'company': nd.get('company_name'),
    'sentiment_score': nd.get('sentiment_score', 0),
    'sentiment_label': nd.get('sentiment_label', 'neutral'),
    'error': nd.get('error')
} for nd in news_data_list], ensure_ascii=False, indent=2)}

## 持仓信息（供参考）
{state.portfolio_summary or "暂无持仓信息"}

⚠️ 重要提示：如果某只股票的数据中包含 "error" 字段（非 null），说明该股票的新闻数据获取失败，请在分析中明确说明：
"[股票代码] 新闻数据获取失败，原因：[error 字段内容]，无法进行情绪分析。"
不要对获取失败的股票编造或推测新闻内容。

请对每只股票分析：
1. 近期新闻的整体情绪倾向（正面/负面/中性）
2. 关键事件或风险点
3. 舆论对股价的潜在影响

请用简洁的中文输出，每只股票一段。"""

    response = await model.ainvoke([HumanMessage(content=prompt)])
    sentiment_summary = response.content if isinstance(response.content, str) else str(response.content)

    logger.info(f"[NewsSentiment] Sentiment analysis completed for {symbols}")
    return {"news_sentiment": sentiment_summary}


# ---------------------------------------------------------------------------
# Sub-Agent 4: Fundamental Analysis（基本面分析节点）
# ---------------------------------------------------------------------------

async def fundamental_analysis_node(
    state: StockAnalysisState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """基本面分析节点：获取财务指标、估值数据、行业信息，并搜索财经深度报道。

    Args:
        state: 当前状态，需要 symbols_to_analyze 字段。
        runtime: LangGraph 运行时。

    Returns:
        更新 fundamental_analysis 字段的状态 patch。
    """
    symbols = state.symbols_to_analyze or []
    if not symbols:
        return {"fundamental_analysis": "无持仓股票，跳过基本面分析。"}

    logger.info(f"[FundamentalAnalysis] Fetching fundamental data for {symbols}")

    # 从持仓中获取公司名称
    portfolio = await async_load_portfolio()

    # 并发获取基本面数据和财经新闻
    import asyncio

    fundamental_tasks = []
    financial_news_tasks = []
    for symbol in symbols:
        pos = portfolio.positions.get(symbol)
        company_name = pos.company_name if pos else None
        fundamental_tasks.append(_get_fundamental_data(symbol, company_name))
        financial_news_tasks.append(_search_financial_news(symbol, company_name))

    fundamental_results = await asyncio.gather(*fundamental_tasks, return_exceptions=True)
    financial_news_results = await asyncio.gather(*financial_news_tasks, return_exceptions=True)

    # 整理数据
    fundamental_data_list = []
    for symbol, result in zip(symbols, fundamental_results):
        if isinstance(result, Exception):
            error_msg = f"{type(result).__name__}: {result}"
            logger.warning(f"[FundamentalAnalysis] Failed to get fundamental data for {symbol}: {error_msg}")
            fundamental_data_list.append({"symbol": symbol, "error": error_msg, "error_type": type(result).__name__})
        else:
            fundamental_data_list.append(result)

    financial_news_list = []
    for symbol, result in zip(symbols, financial_news_results):
        if isinstance(result, Exception):
            error_msg = f"{type(result).__name__}: {result}"
            logger.warning(f"[FundamentalAnalysis] Failed to get financial news for {symbol}: {error_msg}")
            financial_news_list.append({"symbol": symbol, "error": error_msg, "error_type": type(result).__name__})
        else:
            financial_news_list.append(result)

    # 用 LLM 生成基本面分析
    model = load_chat_model(runtime.context.model)
    prompt = f"""你是一位专业的基本面分析师。请根据以下财务数据和财经新闻，对各股票进行基本面分析。

## 财务指标数据
{json.dumps(fundamental_data_list, ensure_ascii=False, indent=2)}

## 财经新闻摘要
{json.dumps([{
    'symbol': fn.get('symbol'),
    'company': fn.get('company_name'),
    'headlines': fn.get('financial_headlines', [])[:4],
    'error': fn.get('error')
} for fn in financial_news_list], ensure_ascii=False, indent=2)}

## 持仓信息（供参考）
{state.portfolio_summary or "暂无持仓信息"}

⚠️ 重要提示：如果某只股票的数据中包含 "error" 字段（非 null），说明该数据获取失败，请在分析中明确说明：
"[股票代码] [数据类型]获取失败，原因：[error 字段内容]，该部分分析无法完成。"
不要对获取失败的数据编造或推测内容。

请对每只股票分析：
1. 估值水平（PE/PB 是否合理，与行业对比）
2. 盈利能力（ROE、利润率、增长趋势）
3. 财务健康度（负债率、流动比率）
4. 分析师评级与目标价
5. 近期财经动态对基本面的影响

请用简洁的中文输出，每只股票一段。"""

    response = await model.ainvoke([HumanMessage(content=prompt)])
    analysis = response.content if isinstance(response.content, str) else str(response.content)

    logger.info(f"[FundamentalAnalysis] Analysis completed for {symbols}")
    return {"fundamental_analysis": analysis}


# ---------------------------------------------------------------------------
# Sub-Agent 5: Stock Recommender（选股推荐节点）
# ---------------------------------------------------------------------------

async def stock_recommender_node(
    state: StockAnalysisState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """选股推荐节点：根据用户预算，从 A 股市场推荐 5 支股票，并给出每支的买入数量和推荐理由。

    流程：
    1. 并发拉取热门 A 股行情（akshare 沪深 A 股实时行情）
    2. 并发拉取候选股票的基本面数据和近期新闻
    3. 用 LLM 综合分析，按预算分配买入数量，输出推荐报告

    Args:
        state: 当前状态，需要 budget 字段（用户预算，单位：元）。
        runtime: LangGraph 运行时。

    Returns:
        更新 recommend_result 和 messages 字段的状态 patch。
    """
    budget = state.budget or 0
    if budget <= 0:
        msg = "❌ 未能识别到有效预算金额，请重新描述，例如：'我有 5 万元，帮我推荐 5 支股票'"
        return {"recommend_result": msg, "messages": [AIMessage(content=msg)]}

    logger.info(f"[StockRecommender] Budget: {budget} CNY, starting recommendation")

    import asyncio

    # ------------------------------------------------------------------
    # Step 1：从 akshare 拉取沪深 A 股实时行情，筛选候选股票
    # ------------------------------------------------------------------
    candidate_symbols: list[str] = []
    candidate_info: list[dict] = []
    try:
        import akshare as ak  # type: ignore

        df = await _ak_call(ak.stock_zh_a_spot_em)
        # 过滤条件：
        # - 价格在预算的 1/50 以内（单手 100 股能买得起）
        # - 成交额 > 5 亿（流动性充足）
        # - 涨跌幅在 -5% ~ +5%（排除涨跌停）
        max_price = budget / 100  # 至少能买 1 手（100 股）
        filtered = df[
            (df["最新价"] > 0)
            & (df["最新价"] <= max_price)
            & (df["成交额"] >= 5e8)
            & (df["涨跌幅"] > -5)
            & (df["涨跌幅"] < 5)
        ].copy()

        # 按成交额降序，取前 30 支作为候选池
        filtered = filtered.sort_values("成交额", ascending=False).head(30)

        for _, row in filtered.iterrows():
            candidate_symbols.append(str(row["代码"]))
            candidate_info.append({
                "symbol": str(row["代码"]),
                "name": str(row["名称"]),
                "price": round(float(row["最新价"]), 2),
                "change_pct": round(float(row["涨跌幅"]), 2),
                "volume_amount": float(row["成交额"]),
                "pe_ratio": float(row.get("市盈率-动态", 0) or 0),
                "pb_ratio": float(row.get("市净率", 0) or 0),
                "market_cap": float(row.get("总市值", 0) or 0),
            })
    except Exception as e:
        fetch_error = f"{type(e).__name__}: {e}"
        logger.warning(f"[StockRecommender] akshare spot fetch failed: {fetch_error}, using fallback list")
        # 兜底：使用一组知名蓝筹股，同时记录错误信息供 LLM 感知
        candidate_symbols = ["600519", "000858", "601318", "600036", "000333",
                              "002415", "600276", "601166", "000002", "600900"]
        candidate_info = [{"symbol": s, "name": s, "data_source_error": fetch_error} for s in candidate_symbols]

    # ------------------------------------------------------------------
    # Step 2：并发拉取候选股票的基本面数据和近期新闻（取前 15 支）
    # ------------------------------------------------------------------
    top_candidates = candidate_symbols[:15]
    top_info = candidate_info[:15]

    fundamental_tasks = [
        _get_fundamental_data(sym, next((c["name"] for c in top_info if c["symbol"] == sym), None))
        for sym in top_candidates
    ]
    news_tasks = [
        _search_stock_news(sym, next((c["name"] for c in top_info if c["symbol"] == sym), None))
        for sym in top_candidates
    ]

    fundamental_results, news_results = await asyncio.gather(
        asyncio.gather(*fundamental_tasks, return_exceptions=True),
        asyncio.gather(*news_tasks, return_exceptions=True),
    )

    # 整理数据
    enriched_candidates = []
    for i, sym in enumerate(top_candidates):
        info = top_info[i].copy()
        fund = fundamental_results[i]
        news = news_results[i]
        if isinstance(fund, Exception):
            error_msg = f"{type(fund).__name__}: {fund}"
            logger.warning(f"[StockRecommender] Failed to get fundamental data for {sym}: {error_msg}")
            info["fundamental_error"] = error_msg
        else:
            info.update(fund)  # type: ignore[arg-type]
        if isinstance(news, Exception):
            error_msg = f"{type(news).__name__}: {news}"
            logger.warning(f"[StockRecommender] Failed to get news for {sym}: {error_msg}")
            info["news_error"] = error_msg
        else:
            info["recent_headlines"] = news.get("headlines", [])[:3]  # type: ignore[union-attr]
        enriched_candidates.append(info)

    # ------------------------------------------------------------------
    # Step 3：LLM 综合分析，按预算分配买入数量，输出推荐报告
    # ------------------------------------------------------------------
    model = load_chat_model(runtime.context.model)

    recommend_prompt = f"""你是一位专业的 A 股投资顾问。用户有 {budget:,.0f} 元的投资预算，希望你从以下候选股票中推荐 5 支，并告知每支应买多少股（以手为单位，1手=100股）。

## 候选股票数据（已按成交额筛选，流动性充足）
{json.dumps(enriched_candidates, ensure_ascii=False, indent=2)}

## 推荐要求
1. 从候选列表中选出最值得买入的 **5 支股票**
2. 预算总计 {budget:,.0f} 元，请合理分配到 5 支股票（可以不均等分配）
3. 每支股票的买入数量必须是 100 的整数倍（整手买入）
4. 确保每支股票的买入金额 = 股价 × 股数，且 5 支合计不超过预算
5. 推荐理由要结合：估值水平（PE/PB）、盈利能力（ROE）、近期新闻情绪、技术面走势

## 输出格式（严格按此格式）

### 📈 选股推荐报告
**投资预算：{budget:,.0f} 元**

---

**第1支：[股票名称]（[代码]）**
- 建议买入：XXX 股（X 手）
- 预计花费：约 X,XXX 元
- 当前价格：XX.XX 元
- 推荐理由：
  1. 估值：（PE/PB 分析）
  2. 基本面：（ROE、营收增长等）
  3. 近期动态：（新闻/事件驱动）
  4. 技术面：（趋势判断）

（以此格式列出全部 5 支）

---

### 💰 预算分配汇总
| 股票 | 代码 | 股数 | 单价 | 花费 |
|------|------|------|------|------|
| ... | ... | ... | ... | ... |
**合计花费：X,XXX 元 | 剩余预算：X,XXX 元**

---

### ⚠️ 风险提示
（简要说明投资风险，2-3句话）

请确保推荐具体可操作，数字计算准确。"""

    response = await model.ainvoke([HumanMessage(content=recommend_prompt)])
    recommendation = response.content if isinstance(response.content, str) else str(response.content)

    logger.info(f"[StockRecommender] Recommendation generated for budget={budget}")
    return {
        "recommend_result": recommendation,
        "messages": [AIMessage(content=recommendation)],
    }



# ---------------------------------------------------------------------------
# Sub-Agent 6: QA Agent（问答分析节点 - ReAct 架构，同时处理追问）
# ---------------------------------------------------------------------------

async def qa_agent_node(
    state: StockAnalysisState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """问答分析节点（ReAct 架构）：按需调用工具回答用户问题，同时兼容追问场景。

    使用 LangGraph 内置的 create_react_agent，无需手写 ReAct 循环：
    - 有工具需求时：LLM 自动推理 -> 调用工具 -> 观察结果 -> 继续推理或给出答案
    - 无工具需求时（如追问）：LLM 直接基于对话历史回答，不触发任何工具调用

    Args:
        state: 当前状态，包含完整对话历史。
        runtime: LangGraph 运行时。

    Returns:
        更新 qa_result 和 messages 字段的状态 patch。
    """
    logger.info("[QAAgent] Starting ReAct QA agent")

    model = load_chat_model(runtime.context.model)

    qa_system_prompt = (
        "你是一位专业的 A 股投资顾问，擅长股票分析、行情解读和投资建议。\n"
        "可用工具：\n"
        "- get_stock_price：获取股票实时行情\n"
        "- get_stock_history：获取历史数据和技术指标（MA、RSI、MACD）\n"
        "- search_stock_news：搜索股票最新新闻和情绪评分\n"
        "- get_fundamental_data：获取基本面数据（PE、PB、ROE等）\n"
        "- search_financial_news：搜索财经深度报道和政策动态\n"
        "- search_stock_by_name：通过公司名称查询股票代码\n"
        "- get_portfolio_info_tool：获取用户当前持仓信息\n\n"
        "工作原则：\n"
        "1. 按需调用工具，不要调用不必要的工具；如果根据对话历史已有足够信息，直接回答即可\n"
        "2. 用户提到公司名称但没有代码时，先用 search_stock_by_name 查询\n"
        "3. 结合完整对话历史理解用户意图（包括追问场景）\n"
        "4. 回答要专业、简洁，涉及投资建议时务必附上风险提示"
    )

    # create_react_agent 内置完整的 ReAct 循环（工具调用 + 结果观察 + 迭代）
    agent = create_react_agent(
        model=model,
        tools=STOKE_BASE_TOOLS,
        prompt=qa_system_prompt,
    )

    # 取最近 20 条历史消息传入，避免 token 超限
    max_history = 20
    history = state.messages[-max_history:] if len(state.messages) > max_history else state.messages

    result = await agent.ainvoke({"messages": list(history)})

    # create_react_agent 返回的 messages 列表中，最后一条是最终回答
    final_msg = result["messages"][-1]
    final_answer = final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content)

    logger.info(f"[QAAgent] QA completed, answer length: {len(final_answer)}")
    return {
        "qa_result": final_answer,
        "messages": [AIMessage(content=final_answer)],
    }
