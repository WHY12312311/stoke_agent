"""持仓管理模块：解析口语化买卖描述，持久化持仓数据。

功能：
- 从自然语言中提取股票代码、买卖方向、数量、价格
- 将持仓数据持久化到本地 JSON 文件
- 提供持仓查询接口
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 持仓数据默认存储路径（可通过环境变量覆盖）
PORTFOLIO_FILE = Path(os.environ.get("PORTFOLIO_FILE", "portfolio.json"))


@dataclass
class Position:
    """单只股票的持仓信息。"""

    symbol: str
    """股票代码，例如 '600519' 或 'AAPL'。"""

    company_name: str = ""
    """公司名称，例如 '贵州茅台'。"""

    shares: float = 0.0
    """持仓股数（手数 * 100 = 股数，A 股 1 手 = 100 股）。"""

    avg_cost: float = 0.0
    """平均持仓成本（元/股）。"""

    total_cost: float = 0.0
    """总持仓成本（元）。"""

    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    """最后更新时间。"""


@dataclass
class Portfolio:
    """整体持仓组合。"""

    positions: dict[str, Position] = field(default_factory=dict)
    """以股票代码为 key 的持仓字典。"""

    trade_history: list[dict] = field(default_factory=list)
    """交易历史记录。"""

    cash_balance: float = 0.0
    """账户现金余额（元）。买入扣减、卖出回收、可直接入金/出金。"""

    def to_dict(self) -> dict:
        return {
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "trade_history": self.trade_history,
            "cash_balance": self.cash_balance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Portfolio":
        portfolio = cls()
        for symbol, pos_data in data.get("positions", {}).items():
            portfolio.positions[symbol] = Position(**pos_data)
        portfolio.trade_history = data.get("trade_history", [])
        portfolio.cash_balance = float(data.get("cash_balance", 0.0))
        return portfolio


def _load_portfolio_sync() -> Portfolio:
    """同步加载持仓数据（内部使用）。

    处理三种情况：
    - 文件不存在：静默返回空持仓
    - 文件为空：说明上次写入未完成，返回空持仓
    - 文件内容损坏：备份原文件后返回空持仓，避免数据丢失
    """
    if not PORTFOLIO_FILE.exists():
        return Portfolio()

    try:
        content = PORTFOLIO_FILE.read_text(encoding="utf-8").strip()
        if not content:
            # 文件存在但为空，说明上次写入未完成，直接重置
            logger.warning(f"[Portfolio] {PORTFOLIO_FILE} is empty, starting fresh")
            return Portfolio()

        data = json.loads(content)
        logger.info(f"[Portfolio] Loaded portfolio from {PORTFOLIO_FILE}")
        return Portfolio.from_dict(data)
    except json.JSONDecodeError as e:
        # 文件内容损坏，备份后重置，避免数据永久丢失
        backup = PORTFOLIO_FILE.with_suffix(".json.bak")
        try:
            PORTFOLIO_FILE.rename(backup)
            logger.warning(
                f"[Portfolio] Corrupted portfolio file (JSONDecodeError: {e}), "
                f"backed up to {backup}, starting fresh"
            )
        except Exception as backup_err:
            logger.warning(
                f"[Portfolio] Corrupted portfolio file (JSONDecodeError: {e}), "
                f"backup failed: {backup_err}, starting fresh"
            )
        return Portfolio()
    except Exception as e:
        logger.warning(f"[Portfolio] Failed to load portfolio: {type(e).__name__}: {e}, starting fresh")
        return Portfolio()


def _save_portfolio_sync(portfolio: Portfolio) -> None:
    """同步保存持仓数据（内部使用）。"""
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(portfolio.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"[Portfolio] Saved portfolio to {PORTFOLIO_FILE}")
    except Exception as e:
        logger.error(f"[Portfolio] Failed to save portfolio: {e}")


def load_portfolio() -> Portfolio:
    """从本地文件加载持仓数据，文件不存在则返回空持仓（同步版本，供非异步场景使用）。"""
    return _load_portfolio_sync()


def save_portfolio(portfolio: Portfolio) -> None:
    """将持仓数据持久化到本地文件（同步版本，供非异步场景使用）。"""
    _save_portfolio_sync(portfolio)


async def async_load_portfolio() -> Portfolio:
    """异步加载持仓数据，避免在 async 环境中阻塞事件循环。"""
    return await asyncio.to_thread(_load_portfolio_sync)


async def async_save_portfolio(portfolio: Portfolio) -> None:
    """异步保存持仓数据，避免在 async 环境中阻塞事件循环。"""
    await asyncio.to_thread(_save_portfolio_sync, portfolio)


def apply_trade(
    portfolio: Portfolio,
    symbol: str,
    company_name: str,
    action: str,  # "buy" 或 "sell"
    shares: float,
    price: float,
) -> str:
    """将一笔交易应用到持仓，返回操作结果描述。

    Args:
        portfolio: 当前持仓对象（会被原地修改）。
        symbol: 股票代码。
        company_name: 公司名称。
        action: 'buy' 或 'sell'。
        shares: 交易股数。
        price: 成交价格（元/股）。

    Returns:
        操作结果的文字描述。
    """
    now = datetime.now().isoformat()

    # 记录交易历史
    trade_record = {
        "time": now,
        "symbol": symbol,
        "company_name": company_name,
        "action": action,
        "shares": shares,
        "price": price,
        "amount": round(shares * price, 2),
    }
    portfolio.trade_history.append(trade_record)

    trade_amount = round(shares * price, 2)

    if action == "buy":
        # Check if there is enough cash (allow buying even without sufficient cash, but warn)
        cash_warning = ""
        if portfolio.cash_balance < trade_amount:
            shortfall = trade_amount - portfolio.cash_balance
            cash_warning = f"\n⚠️ 账户现金不足（缺口 {shortfall:.2f} 元），已透支处理。建议后续入金补足。"

        portfolio.cash_balance = round(portfolio.cash_balance - trade_amount, 2)

        if symbol in portfolio.positions:
            pos = portfolio.positions[symbol]
            # Weighted average cost
            total_shares = pos.shares + shares
            total_cost = pos.total_cost + trade_amount
            pos.shares = total_shares
            pos.avg_cost = round(total_cost / total_shares, 4)
            pos.total_cost = round(total_cost, 2)
            pos.last_updated = now
            if company_name:
                pos.company_name = company_name
        else:
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                company_name=company_name,
                shares=shares,
                avg_cost=round(price, 4),
                total_cost=trade_amount,
                last_updated=now,
            )
        result = (
            f"✅ 买入 {company_name or symbol} {shares:.0f} 股，"
            f"成交价 {price:.2f} 元，总金额 {trade_amount:.2f} 元"
            f"{cash_warning}"
        )

    elif action == "sell":
        if symbol not in portfolio.positions or portfolio.positions[symbol].shares <= 0:
            return f"❌ 卖出失败：{company_name or symbol} 当前无持仓"

        pos = portfolio.positions[symbol]
        if shares > pos.shares:
            return f"❌ 卖出失败：持仓不足，当前持有 {pos.shares:.0f} 股，尝试卖出 {shares:.0f} 股"

        # Sell proceeds go back to cash balance
        portfolio.cash_balance = round(portfolio.cash_balance + trade_amount, 2)

        pos.shares -= shares
        pos.total_cost = round(pos.avg_cost * pos.shares, 2)
        pos.last_updated = now

        # Remove position if fully sold
        if pos.shares <= 0:
            del portfolio.positions[symbol]
            result = f"✅ 卖出 {company_name or symbol} {shares:.0f} 股，成交价 {price:.2f} 元，回收 {trade_amount:.2f} 元，已清仓"
        else:
            result = f"✅ 卖出 {company_name or symbol} {shares:.0f} 股，成交价 {price:.2f} 元，回收 {trade_amount:.2f} 元，剩余 {pos.shares:.0f} 股"
    else:
        result = f"❌ 未知操作类型：{action}"

    return result


def deposit_cash(portfolio: Portfolio, amount: float) -> str:
    """向账户入金。

    Args:
        portfolio: 当前持仓对象（会被原地修改）。
        amount: 入金金额（元），必须大于 0。

    Returns:
        操作结果描述。
    """
    if amount <= 0:
        return f"❌ 入金金额必须大于 0，收到 {amount:.2f} 元"

    portfolio.cash_balance = round(portfolio.cash_balance + amount, 2)
    portfolio.trade_history.append({
        "time": datetime.now().isoformat(),
        "action": "deposit",
        "amount": amount,
    })
    return f"✅ 入金 {amount:,.2f} 元，当前账户现金余额：{portfolio.cash_balance:,.2f} 元"


def withdraw_cash(portfolio: Portfolio, amount: float) -> str:
    """从账户出金。

    Args:
        portfolio: 当前持仓对象（会被原地修改）。
        amount: 出金金额（元），必须大于 0 且不超过现金余额。

    Returns:
        操作结果描述。
    """
    if amount <= 0:
        return f"❌ 出金金额必须大于 0，收到 {amount:.2f} 元"
    if amount > portfolio.cash_balance:
        return f"❌ 出金失败：当前现金余额 {portfolio.cash_balance:,.2f} 元，不足以出金 {amount:,.2f} 元"

    portfolio.cash_balance = round(portfolio.cash_balance - amount, 2)
    portfolio.trade_history.append({
        "time": datetime.now().isoformat(),
        "action": "withdraw",
        "amount": amount,
    })
    return f"✅ 出金 {amount:,.2f} 元，当前账户现金余额：{portfolio.cash_balance:,.2f} 元"


def get_portfolio_summary(portfolio: Portfolio) -> str:
    """生成持仓摘要文本，供 LLM 分析使用。"""
    lines = []

    # Cash balance section (always shown)
    lines.append(f"💵 账户现金余额：{portfolio.cash_balance:,.2f} 元\n")

    if not portfolio.positions:
        lines.append("📋 当前无股票持仓。")
        total_assets = portfolio.cash_balance
        lines.append(f"\n💰 账户总资产（现金）：{total_assets:,.2f} 元")
        return "\n".join(lines)

    lines.append("📋 当前持仓明细：\n")
    total_cost = 0.0
    for symbol, pos in portfolio.positions.items():
        lines.append(
            f"  • {pos.company_name or symbol}（{symbol}）"
            f"  持仓 {pos.shares:.0f} 股"
            f"  | 均价 {pos.avg_cost:.2f} 元"
            f"  | 持仓成本 {pos.total_cost:.2f} 元"
        )
        total_cost += pos.total_cost

    lines.append(f"\n💰 总持仓成本：{total_cost:,.2f} 元")
    lines.append(f"💵 账户现金余额：{portfolio.cash_balance:,.2f} 元")
    lines.append(f"💰 账户总资产（持仓成本+现金）：{total_cost + portfolio.cash_balance:,.2f} 元")
    lines.append(f"📊 持仓股票数：{len(portfolio.positions)} 只")
    return "\n".join(lines)
