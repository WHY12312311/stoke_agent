"""股票多智能体系统。

基于 Supervisor 架构的多智能体股票分析系统，支持两种模式：
1. 交易记录模式：解析口语化买卖描述，持久化持仓数据
2. 持仓分析模式：对所有持仓股票进行行情、技术、新闻、基本面四维分析

子 Agent：
- TradeRecorder         : 口语化交易解析与持仓管理
- MarketMonitorAgent    : 实时行情监控与盈亏分析
- TechnicalAnalysisAgent: MA/RSI/MACD 技术指标分析
- NewsSentimentAgent    : 新闻情绪分析
- FundamentalAnalysisAgent: 财务指标与基本面分析
"""

from stoke_multi_agent.graph import graph

__all__ = ["graph"]
