"""挤压突破 — 布林带-Keltner 挤压后向上突破 + 放量

原理:布林带 BB(20, 2σ) 缩进 Keltner KC(20, mult×ATR14) 内部 = 波动极度萎缩
(挤压),常为变盘前兆;近期发生过挤压且价格向上突破布林上轨,视为挤压向上释放、
变盘方向确认。配合放量 / 中期趋势过滤提高胜率。

与 boll_breakout 的区别:多了"突破前必须先经历波动挤压"这一前提,过滤掉没有能量
积累的普通突破(对齐个股分析页的挤压徽标口径:BB(20,2) 缩进 KC(20,1.5×ATR))。
矩阵无预计算 ATR 列,故 ATR14 在此现场用真实波幅(TR)滚动均值算出。
"""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
    valid_rolling_mean,
    valid_shift,
)

META = {
    "id": "squeeze_breakout",
    "name": "挤压突破",
    "description": "布林带缩进 Keltner(波动挤压)后向上突破布林上轨 + 放量",
    "tags": ["挤压", "布林", "Keltner", "突破"],
    "asset_types": ["stock", "etf"],
    "timeframes": ["1d"],
    "params": [
        {
            "id": "squeeze_atr_mult",
            "label": "Keltner ATR 倍数",
            "type": "float",
            "default": 1.5,
            "min": 1.0,
            "max": 2.5,
            "step": 0.1,
        },
        {
            "id": "release_lookback",
            "label": "挤压回看天数",
            "type": "int",
            "default": 3,
            "min": 1,
            "max": 10,
        },
        {
            "id": "require_upper_breakout",
            "label": "要求突破布林上轨",
            "type": "bool",
            "default": True,
        },
        {"id": "use_volume_filter", "label": "启用放量过滤", "type": "bool", "default": True},
        {
            "id": "vol_ratio_min",
            "label": "最低量比",
            "type": "float",
            "default": 1.2,
            "min": 0.5,
            "max": 5.0,
            "step": 0.1,
        },
        {
            "id": "require_trend_up",
            "label": "要求收盘价在MA60上方",
            "type": "bool",
            "default": False,
        },
    ],
    "scoring": {"vol_ratio_5d": 0.4, "momentum_20d": 0.3, "change_pct": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_boll_breakout_upper"]
EXIT_SIGNALS = ["signal_ma20_breakdown"]
STOP_LOSS = -0.06
MAX_HOLD_DAYS = 15
ALERTS = []


class SqueezeBreakoutMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        # high/low/open 为核心 OHLCV, 恒被加载; 这里声明需现场用到的基础列
        return frozenset({"close", "volume"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 60

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        close = market.close
        ma20 = matrix_feature(market, "ma20")
        upper = matrix_feature(market, "boll_upper")   # MA20 + 2σ
        lower = matrix_feature(market, "boll_lower")   # MA20 - 2σ

        # —— Keltner 通道: MA20 ± mult×ATR14 ——
        # 矩阵无预计算 ATR 列, 现场从真实波幅 TR 算:
        #   TR = max(high-low, |high-prev_close|, |low-prev_close|)
        #   ATR14 = TR 的 14 日滚动均值(用 valid_rolling_mean 跳过停牌缺口)
        mult = float(params.get("squeeze_atr_mult", 1.5))
        prev_close = valid_shift(close, 1)
        tr = np.maximum(
            market.high - market.low,
            np.maximum(
                np.abs(market.high - prev_close),
                np.abs(market.low - prev_close),
            ),
        )
        atr = valid_rolling_mean(tr, np.isfinite(tr), 14)
        kc_up = ma20 + mult * atr
        kc_lo = ma20 - mult * atr

        # —— 挤压判定: 布林带完全缩进 Keltner 内部(NaN 比较得 False, 自动过滤热身期)——
        squeeze_on = (upper < kc_up) & (lower > kc_lo)

        # —— 近 N 根内发生过挤压(含当根): 允许突破在挤压释放后 1~N 日内触发 ——
        lookback = int(params.get("release_lookback", 3))
        squeeze_f = squeeze_on.astype(np.float32)
        recent = squeeze_on.copy()
        for k in range(1, lookback + 1):
            recent |= valid_shift(squeeze_f, k) >= 0.5

        entry = recent.copy()
        if params.get("require_upper_breakout", True):
            entry &= close > upper                       # 向上突破布林上轨 = 挤压向上释放
        if params.get("use_volume_filter", True):
            entry &= matrix_feature(market, "vol_ratio_5d") >= float(
                params.get("vol_ratio_min", 1.2)
            )
        if params.get("require_trend_up", False):
            entry &= close > matrix_feature(market, "ma60")

        exit_ = close < ma20                             # 跌破 MA20 = 突破失效

        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            exit=exit_.astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(exit_, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_boll_breakout_upper",),
            exit_signal_ids=("signal_ma20_breakdown",),
        )


MATRIX_STRATEGY = SqueezeBreakoutMatrixStrategy()
