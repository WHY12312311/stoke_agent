"""多智能体系统的股票分析工具集。

提供四类工具：
1. 行情数据工具  - 获取实时价格、成交量等
2. 技术分析工具  - 计算 MA、RSI、MACD 等指标
3. 新闻情绪工具  - 搜索并评分最新新闻
4. 基本面工具    - 获取财务指标、行业信息、估值数据
"""

import asyncio
import logging
from typing import Any, Optional

import akshare as ak  # type: ignore
from langchain_core.tools import tool
from stoke_multi_agent.portfolio import (
    apply_trade,
    async_load_portfolio,
    async_save_portfolio,
    get_portfolio_summary,
    load_portfolio,
    Portfolio,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global concurrency limiter & retry helper for akshare calls.
# akshare uses synchronous HTTP (requests) under the hood; we push each call
# to a thread via asyncio.to_thread().  The semaphore prevents overwhelming
# the upstream data source with too many simultaneous connections.
# ---------------------------------------------------------------------------
_AK_SEMAPHORE = asyncio.Semaphore(5)  # max 5 concurrent akshare requests
_AK_MAX_RETRIES = 3
_AK_RETRY_BASE_DELAY = 1.0  # seconds, exponential backoff


async def _ak_call(func, *args, **kwargs):
    """Run a blocking akshare function in a thread with concurrency limiting and retry.

    - Acquires a semaphore slot to cap concurrent requests.
    - Retries up to _AK_MAX_RETRIES times on connection errors with exponential backoff.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _AK_MAX_RETRIES + 1):
        async with _AK_SEMAPHORE:
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as exc:
                last_exc = exc
                # Only retry on connection-level errors
                exc_msg = str(exc).lower()
                retriable = any(kw in exc_msg for kw in (
                    "remotedisconnected", "connection aborted", "connectionreset",
                    "connectionrefused", "timeout", "timed out", "blocking",
                ))
                if not retriable or attempt == _AK_MAX_RETRIES:
                    raise
                delay = _AK_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[akshare] {func.__name__} attempt {attempt}/{_AK_MAX_RETRIES} failed: {exc}, "
                    f"retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
    # Should not reach here, but just in case
    raise last_exc  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Market Data Tools
# ---------------------------------------------------------------------------

async def _get_stock_price(symbol: str) -> dict[str, Any]:
    """获取指定 A 股股票代码的最新行情数据。

    Args:
        symbol: A 股股票代码，例如 '600519'（贵州茅台）。

    Returns:
        包含 price、change_pct、volume 等字段的字典。
    """
    try:
        from datetime import date, timedelta

        # 获取最近 5 天的日线数据（包含今天），避免拉取全市场数据
        end_date = date.today().strftime("%Y%m%d")
        start_date = (date.today() - timedelta(days=7)).strftime("%Y%m%d")

        df = await _ak_call(
            ak.stock_zh_a_hist,
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )

        if df.empty:
            raise ValueError(f"Symbol {symbol} not found or no recent data")

        # 取最新一条数据
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        price = float(latest.get("收盘", 0) or 0)
        prev_close = float(prev.get("收盘", 0) or price)
        change_amount = price - prev_close
        change_pct = (change_amount / prev_close * 100) if prev_close > 0 else 0.0

        return {
            "symbol": symbol,
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "change_amount": round(change_amount, 2),
            "volume": int(latest.get("成交量", 0) or 0),
            "turnover": float(latest.get("成交额", 0) or 0),
            "high": round(float(latest.get("最高", 0) or 0), 2),
            "low": round(float(latest.get("最低", 0) or 0), 2),
            "open": round(float(latest.get("开盘", 0) or 0), 2),
            "prev_close": round(prev_close, 2),
            "date": str(latest.get("日期", "")),
            "source": "akshare",
        }
    except Exception as e:
        logger.warning(f"akshare fetch failed for {symbol}: {e}, using mock data")
        return _mock_market_data(symbol)


@tool
async def get_stock_price(symbol: str) -> dict[str, Any]:
    """获取指定 A 股股票代码的最新行情数据，包含价格、涨跌幅、成交量等。
    symbol 必须是6位数字的 A 股代码，例如 '600519'（贵州茅台）。
    """
    return await _get_stock_price(symbol)


# period 到 akshare 天数的映射
_PERIOD_TO_DAYS: dict[str, int] = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
}


async def _get_stock_history(symbol: str, period: str = "1mo") -> dict[str, Any]:
    """获取 A 股历史 OHLCV 数据，用于技术指标计算。

    Args:
        symbol: A 股股票代码，例如 '600519'。
        period: 时间周期，支持 '1mo'、'3mo'、'6mo'、'1y'、'2y'。

    Returns:
        包含 'closes'、'highs'、'lows'、'volumes' 列表的字典。
    """
    try:
        from datetime import date, timedelta

        days = _PERIOD_TO_DAYS.get(period, 90)
        end_date = date.today().strftime("%Y%m%d")
        start_date = (date.today() - timedelta(days=days)).strftime("%Y%m%d")

        # adjust="qfq" 表示前复权，适合技术分析
        df = await _ak_call(
            ak.stock_zh_a_hist,
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )

        if df.empty:
            raise ValueError(f"No history data for {symbol}")

        return {
            "symbol": symbol,
            "closes": df["收盘"].tolist(),
            "highs": df["最高"].tolist(),
            "lows": df["最低"].tolist(),
            "volumes": df["成交量"].tolist(),
            "dates": df["日期"].astype(str).tolist(),
        }
    except Exception as e:
        logger.warning(f"akshare history fetch failed for {symbol}: {e}, using mock data")
        return _mock_history_data(symbol)


@tool
async def get_stock_history(symbol: str, period: str = "1mo") -> dict[str, Any]:
    """获取 A 股历史数据并计算技术指标（MA5/MA20、RSI14、MACD）。
    symbol: 6位 A 股代码。
    period: 时间周期，支持 '1mo'、'3mo'、'6mo'、'1y'、'2y'。
    """
    return await _get_stock_history(symbol, period)


# ---------------------------------------------------------------------------
# Technical Analysis Tools
# ---------------------------------------------------------------------------

def compute_technical_indicators(history: dict[str, Any]) -> dict[str, Any]:
    """根据历史价格数据计算常用技术指标。

    计算内容：
    - MA5、MA20（移动平均线）
    - RSI14（相对强弱指数）
    - MACD（12/26/9 EMA）

    Args:
        history: get_stock_history() 的返回结果。

    Returns:
        包含各指标数值及简单信号标签的字典。
    """
    closes = history.get("closes", [])
    if len(closes) < 26:
        return {"error": "Insufficient data for technical analysis (need >= 26 days)"}

    # --- Moving Averages ---
    ma5 = _sma(closes, 5)
    ma20 = _sma(closes, 20)
    ma_signal = "bullish" if ma5 > ma20 else "bearish"

    # --- RSI ---
    rsi = _rsi(closes, 14)
    if rsi >= 70:
        rsi_signal = "overbought"
    elif rsi <= 30:
        rsi_signal = "oversold"
    else:
        rsi_signal = "neutral"

    # --- MACD ---
    macd_line, signal_line, histogram = _macd(closes)
    macd_signal = "bullish" if histogram > 0 else "bearish"

    return {
        "symbol": history.get("symbol", ""),
        "ma5": round(ma5, 2),
        "ma20": round(ma20, 2),
        "ma_signal": ma_signal,
        "rsi14": round(rsi, 2),
        "rsi_signal": rsi_signal,
        "macd_line": round(macd_line, 4),
        "signal_line": round(signal_line, 4),
        "macd_histogram": round(histogram, 4),
        "macd_signal": macd_signal,
    }


# ---------------------------------------------------------------------------
# News Sentiment Tools
# ---------------------------------------------------------------------------

async def _search_stock_news(symbol: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """搜索指定股票的最新新闻并计算情绪评分。

    Args:
        symbol: 股票代码。
        company_name: 可选的公司名称，有助于提升搜索精度。

    Returns:
        包含新闻标题列表及综合情绪评分（-1 到 1）的字典。
    """
    try:
        from langchain_tavily import TavilySearch  # type: ignore

        query = f"{company_name or symbol} 股票 最新新闻 财经"
        search = TavilySearch(max_results=5)
        results = await search.ainvoke({"query": query})

        headlines = []
        if isinstance(results, dict) and "results" in results:
            headlines = [r.get("title", "") for r in results["results"] if r.get("title")]
        elif isinstance(results, list):
            headlines = [r.get("title", "") for r in results if isinstance(r, dict)]

        # 简单关键词情绪评分（生产环境可替换为 LLM 评分）
        sentiment_score = _simple_sentiment_score(headlines)

        return {
            "symbol": symbol,
            "company_name": company_name or symbol,
            "headlines": headlines[:5],
            "sentiment_score": sentiment_score,
            "sentiment_label": _score_to_label(sentiment_score),
        }
    except Exception as e:
        logger.warning(f"News search failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "company_name": company_name or symbol,
            "headlines": [],
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "error": str(e),
        }


@tool
async def search_stock_news(symbol: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """搜索指定股票的最新新闻并计算情绪评分。
    symbol: 股票代码。company_name: 可选的公司名称。
    """
    return await _search_stock_news(symbol, company_name)


# ---------------------------------------------------------------------------
# Stock Code Lookup Tools（股票代码查询）
# ---------------------------------------------------------------------------

# 全市场股票列表缓存（避免每次都重新拉取）
_stock_code_cache: dict[str, str] | None = None


async def _search_stock_by_name(name_query: str) -> dict[str, Any]:
    """通过公司名称或简称模糊搜索 A 股股票代码。

    优先使用 akshare 的全市场股票列表做本地模糊匹配，速度快且无需网络请求。
    匹配逻辑：
    1. 精确匹配（完整名称）
    2. 包含匹配（name_query 是股票名称的子串）
    3. 反向包含（股票名称是 name_query 的子串）

    Args:
        name_query: 用户输入的公司名称或简称，例如 "茅台"、"贵州茅台"、"比亚迪"。

    Returns:
        包含匹配结果列表的字典，每条结果含 symbol 和 name 字段。
        示例：{"query": "茅台", "matches": [{"symbol": "600519", "name": "贵州茅台"}]}
    """
    global _stock_code_cache

    try:
        # 首次调用时拉取全市场股票列表并缓存
        if _stock_code_cache is None:
            df = await _ak_call(ak.stock_info_a_code_name)
            # 列名：code（代码）、name（名称）
            _stock_code_cache = dict(zip(df["code"].astype(str), df["name"].astype(str)))
            logger.info(f"[StockLookup] Loaded {len(_stock_code_cache)} A-share stocks into cache")

        query = name_query.strip()
        matches = []

        for code, name in _stock_code_cache.items():
            # 精确匹配 > 包含匹配 > 反向包含
            if name == query:
                matches.insert(0, {"symbol": code, "name": name, "match_type": "exact"})
            elif query in name or name in query:
                matches.append({"symbol": code, "name": name, "match_type": "fuzzy"})

        # 精确匹配优先，最多返回 5 条
        matches = matches[:5]

        return {
            "query": name_query,
            "matches": matches,
            "total": len(matches),
        }

    except Exception as e:
        logger.warning(f"[StockLookup] search_stock_by_name failed for '{name_query}': {e}")
        return {
            "query": name_query,
            "matches": [],
            "total": 0,
            "error": str(e),
        }


@tool
async def search_stock_by_name(name_query: str) -> dict[str, Any]:
    """通过公司名称或简称模糊搜索 A 股股票代码。
    name_query: 用户输入的公司名称或简称，例如 '茅台'、'比亚迪'。
    返回匹配的股票代码和名称列表。
    """
    return await _search_stock_by_name(name_query)


# ---------------------------------------------------------------------------
# Fundamental Analysis Tools（新增）
# ---------------------------------------------------------------------------

async def _get_fundamental_data(symbol: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """获取 A 股基本面数据：财务指标、估值、行业信息。

    数据来源：
    - stock_individual_info_em：基本信息（市值、行业）
    - stock_financial_abstract_ths：ROE、营收、净利润等财务摘要（同花顺）
    - stock_zh_a_spot_em：PE、PB 等实时估值指标

    Args:
        symbol: A 股股票代码，例如 '600519'。
        company_name: 可选的公司名称。

    Returns:
        包含 PE、PB、ROE、营收等基本面指标的字典。
    """
    try:
        result: dict[str, Any] = {
            "symbol": symbol,
            "company_name": company_name or symbol,
            "source": "akshare",
        }

        # --- 基本信息：总市值、行业 ---
        try:
            info_df = await _ak_call(ak.stock_individual_info_em, symbol=symbol)
            if not info_df.empty:
                info_dict = dict(zip(info_df["item"], info_df["value"]))
                result["company_name"] = str(info_dict.get("股票简称", company_name or symbol))
                result["market_cap"] = _safe_float(info_dict.get("总市值"))
                result["industry"] = str(info_dict.get("行业", ""))
                result["list_date"] = str(info_dict.get("上市时间", ""))
        except Exception as e:
            logger.warning(f"stock_individual_info_em failed for {symbol}: {e}")

        # --- 实时估值：PE / PB（百度财经，单只查询，避免拉取全市场数据）---
        try:
            pe_df, pb_df = await asyncio.gather(
                _ak_call(ak.stock_zh_valuation_baidu, symbol=symbol, indicator="市盈率(TTM)"),
                _ak_call(ak.stock_zh_valuation_baidu, symbol=symbol, indicator="市净率"),
            )
            if not pe_df.empty:
                result["pe_ratio"] = _safe_float(pe_df.iloc[-1].get("value"))
            if not pb_df.empty:
                result["pb_ratio"] = _safe_float(pb_df.iloc[-1].get("value"))
        except Exception as e:
            logger.warning(f"stock_zh_valuation_baidu failed for {symbol}: {e}")

        # --- 财务摘要：ROE / 营收 / 净利润（同花顺）---
        try:
            fin_df = await _ak_call(ak.stock_financial_abstract_ths, symbol=symbol, indicator="按年度")
            if not fin_df.empty:
                latest_fin = fin_df.iloc[0]  # 最新一期在第一行
                result["roe"] = _safe_float(latest_fin.get("净资产收益率"))
                result["revenue"] = _safe_float(latest_fin.get("营业总收入"))
                result["net_profit"] = _safe_float(latest_fin.get("净利润"))
                result["revenue_growth"] = _safe_float(latest_fin.get("营业总收入同比增长率"))
                result["earnings_growth"] = _safe_float(latest_fin.get("净利润同比增长率"))
                result["report_date"] = str(latest_fin.get("报告期", ""))
        except Exception as e:
            logger.warning(f"stock_financial_abstract_ths failed for {symbol}: {e}")

        return result
    except Exception as e:
        logger.warning(f"Fundamental data fetch failed for {symbol}: {e}, using mock data")
        return _mock_fundamental_data(symbol, company_name)


@tool
async def get_fundamental_data(symbol: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """获取 A 股基本面数据：PE、PB、ROE、营收、净利润、行业信息等。
    symbol: 6位 A 股代码。company_name: 可选的公司名称。
    """
    return await _get_fundamental_data(symbol, company_name)


async def _search_financial_news(symbol: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """搜索股票相关的财经、舆论、政策新闻（比 search_stock_news 更广泛）。

    Args:
        symbol: 股票代码。
        company_name: 可选的公司名称。

    Returns:
        包含财经新闻、政策动态、行业舆论的字典。
    """
    try:
        from langchain_tavily import TavilySearch  # type: ignore

        name = company_name or symbol
        # 多维度搜索：财经 + 政策 + 行业
        queries = [
            f"{name} 财报 业绩 营收",
            f"{name} 行业政策 监管",
        ]

        all_results = []
        search = TavilySearch(max_results=4)

        for query in queries:
            try:
                results = await search.ainvoke({"query": query})
                if isinstance(results, dict) and "results" in results:
                    all_results.extend(results["results"])
                elif isinstance(results, list):
                    all_results.extend(results)
            except Exception as e:
                logger.warning(f"Sub-query failed for '{query}': {e}")

        # 去重并提取标题
        seen = set()
        headlines = []
        for r in all_results:
            title = r.get("title", "") if isinstance(r, dict) else ""
            if title and title not in seen:
                seen.add(title)
                headlines.append(title)

        return {
            "symbol": symbol,
            "company_name": name,
            "financial_headlines": headlines[:8],
            "total_found": len(headlines),
        }
    except Exception as e:
        logger.warning(f"Financial news search failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "company_name": company_name or symbol,
            "financial_headlines": [],
            "total_found": 0,
            "error": str(e),
        }


@tool
async def search_financial_news(symbol: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """搜索股票相关的财经深度报道、政策动态、行业舆论（比 search_stock_news 更广泛）。
    symbol: 6位 A 股代码。company_name: 可选的公司名称。
    """
    return await _search_financial_news(symbol, company_name)


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> Optional[float]:
    """安全地将值转换为 float，失败时返回 None。"""
    try:
        if val is None or val == "" or val == "--":
            return None
        return float(str(val).replace("%", "").replace(",", ""))
    except (ValueError, TypeError):
        return None


def _sma(data: list[float], period: int) -> float:
    """简单移动平均线（SMA）。"""
    if len(data) < period:
        return data[-1] if data else 0.0
    return sum(data[-period:]) / period


def _ema(data: list[float], period: int) -> float:
    """指数移动平均线（EMA），返回最后一个值。"""
    if not data:
        return 0.0
    k = 2 / (period + 1)
    ema = data[0]
    for price in data[1:]:
        ema = price * k + ema * (1 - k)
    return ema


def _rsi(closes: list[float], period: int = 14) -> float:
    """相对强弱指数（RSI）。"""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains[-period:]) / period if gains else 0
    avg_loss = sum(losses[-period:]) / period if losses else 0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
    """MACD 指标，返回 (macd_line, signal_line, histogram)。"""
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line * (2 / (signal + 1))
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _simple_sentiment_score(headlines: list[str]) -> float:
    """基于关键词的情绪评分，范围 -1.0 到 1.0。"""
    positive_words = {"surge", "rally", "beat", "profit", "growth", "bullish", "buy", "upgrade",
                      "上涨", "涨停", "利好", "增长", "盈利", "买入", "上调", "超预期", "创新高"}
    negative_words = {"crash", "fall", "loss", "bearish", "sell", "downgrade", "risk", "warn",
                      "下跌", "跌停", "利空", "亏损", "卖出", "下调", "风险", "调查", "处罚"}
    score = 0.0
    count = 0
    for headline in headlines:
        text = headline.lower()
        for w in positive_words:
            if w in text:
                score += 1
                count += 1
        for w in negative_words:
            if w in text:
                score -= 1
                count += 1
    if count == 0:
        return 0.0
    return max(-1.0, min(1.0, score / count))


def _score_to_label(score: float) -> str:
    """将数值情绪评分转换为可读标签。"""
    if score > 0.3:
        return "positive"
    elif score < -0.3:
        return "negative"
    return "neutral"


def _mock_market_data(symbol: str) -> dict[str, Any]:
    """返回用于开发/测试的模拟行情数据。"""
    import random
    price = round(random.uniform(10, 500), 2)
    return {
        "symbol": symbol,
        "price": price,
        "change_pct": round(random.uniform(-5, 5), 2),
        "volume": random.randint(1_000_000, 50_000_000),
        "market_cap": round(price * random.randint(100_000_000, 10_000_000_000), 2),
        "source": "mock",
    }


def _mock_fundamental_data(symbol: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """返回用于开发/测试的模拟基本面数据。"""
    import random
    return {
        "symbol": symbol,
        "company_name": company_name or symbol,
        "sector": "Consumer Staples",
        "industry": "Beverages",
        "pe_ratio": round(random.uniform(15, 50), 2),
        "forward_pe": round(random.uniform(12, 40), 2),
        "pb_ratio": round(random.uniform(1, 10), 2),
        "roe": round(random.uniform(0.05, 0.35), 4),
        "revenue_growth": round(random.uniform(-0.1, 0.3), 4),
        "earnings_growth": round(random.uniform(-0.1, 0.4), 4),
        "profit_margin": round(random.uniform(0.05, 0.4), 4),
        "debt_to_equity": round(random.uniform(0, 100), 2),
        "current_ratio": round(random.uniform(0.5, 3), 2),
        "dividend_yield": round(random.uniform(0, 0.05), 4),
        "52w_high": round(random.uniform(200, 600), 2),
        "52w_low": round(random.uniform(50, 200), 2),
        "analyst_target_price": round(random.uniform(100, 500), 2),
        "recommendation": random.choice(["buy", "hold", "sell"]),
        "source": "mock",
    }


def _mock_history_data(symbol: str) -> dict[str, Any]:
    """返回用于开发/测试的模拟历史数据。"""
    import random
    base = random.uniform(50, 300)
    closes = [round(base + random.uniform(-10, 10), 2) for _ in range(60)]
    return {
        "symbol": symbol,
        "closes": closes,
        "highs": [c + random.uniform(0, 5) for c in closes],
        "lows": [c - random.uniform(0, 5) for c in closes],
        "volumes": [random.randint(1_000_000, 10_000_000) for _ in closes],
        "dates": [f"2025-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}" for i in range(60)],
    }



@tool
def get_portfolio_info_tool() -> dict:
    """获取当前用户的持仓信息摘要，包含各股票的持仓数量、成本价、账户现金余额等。无需任何参数。"""
    portfolio = load_portfolio()
    summary = get_portfolio_summary(portfolio)
    total_cost = sum(pos.total_cost for pos in portfolio.positions.values())
    return {
        "summary": summary,
        "positions": {
            symbol: {
                "company_name": pos.company_name,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "total_cost": pos.total_cost,
            }
            for symbol, pos in portfolio.positions.items()
        },
        "total_positions": len(portfolio.positions),
        "cash_balance": portfolio.cash_balance,
        "total_assets": round(total_cost + portfolio.cash_balance, 2),
    }

STOKE_BASE_TOOLS = [
    get_stock_price,
    get_stock_history,
    search_stock_news,
    get_fundamental_data,
    search_financial_news,
    search_stock_by_name,
    get_portfolio_info_tool,
]