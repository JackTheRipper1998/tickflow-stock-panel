#!/usr/bin/env python
"""消融第二轮 (聚焦): 族内最优档组合 + 跟踪止盈口径下的完整验证。

第一轮 (ablation_trend_upgrade.py) 的两个遗留:
  1. 嵌套参数族 (C1/C6/C7) 全部 AND = 只取最严档; 本轮按预定规则
     「族内取笔均最高档, 平手取池子更大档」选一档进组合。
  2. walk-forward 只验了持有1天; 本轮对 破MA10跟踪止盈 (第一轮明显最优的
     持有口径) 做前后分段验证 — 这是新策略定位能否成立的关键。

组合定义 (基于第一轮已固定的消融表, 不重新调参):
  COMBINED = BASE + 多头持续≥7 + MA20斜率>0 + 低点抬高 + 回归R²≥0.65
             + ADX(10)≥30且+DI>-DI + OBV20均线向上
  MINIMAL  = BASE + MA20斜率>0 + 回归斜率>0且R²≥0.60   (需求四: 只允许两条时的优先级)
"""
from __future__ import annotations

import logging

import polars as pl

from scripts.ablation_trend_upgrade import (
    _add_adx_10,
    _load_panel,
    build_trades,
    fmt,
    score_correlation,
    stats,
)

logger = logging.getLogger(__name__)

HOLDS = [("持有1天", "ret_h1"), ("持有3天", "ret_h3"), ("持有5天", "ret_h5"),
         ("破MA10跟踪止盈", "ret_trail")]


def _configs() -> dict[str, pl.Expr]:
    c1 = pl.col("bull_align_days") >= 7
    c2 = pl.col("ma20_slope_5d") > 0
    c4 = pl.col("low_uplift_10d") > 0
    c6_065 = (pl.col("reg_slope_20d") > 0) & (pl.col("reg_r2_20d") >= 0.65)
    c6_060 = (pl.col("reg_slope_20d") > 0) & (pl.col("reg_r2_20d") >= 0.60)
    c7 = (pl.col("adx_10") >= 30) & (pl.col("plus_di_10") > pl.col("minus_di_10"))
    c8b = pl.col("obv_ma20_slope_5d") > 0
    return {
        "BASE": pl.lit(True),
        "MINIMAL(MA20斜率+R²≥0.60)": (c2 & c6_060),
        "COMBINED(族内最优档×6)": (c1 & c2 & c4 & c6_065 & c7 & c8b),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    panel = _load_panel()
    panel = _add_adx_10(panel)
    trades = build_trades(panel)

    didx_list = sorted(trades["didx"].unique().to_list())
    mid = didx_list[len(didx_list) // 2]

    print(f"\n{'='*100}")
    print("第二轮: 三种配置 × 四种持有口径 (全部净收益, 已扣费0.25%)")
    print(f"{'='*100}")

    for cfg_name, mask in _configs().items():
        sub = trades.filter(mask.fill_null(False))
        print(f"\n── {cfg_name} ──")
        for label, col in HOLDS:
            print(fmt(f"{label}", stats(sub, col)))

    print(f"\n{'='*100}")
    print("walk-forward 前后分段 (跟踪止盈 + 持有1天 两个口径)")
    print(f"{'='*100}")
    for cfg_name, mask in _configs().items():
        sub = trades.filter(mask.fill_null(False))
        for seg_label, seg in [("前半", sub.filter(pl.col("didx") < mid)),
                               ("后半", sub.filter(pl.col("didx") >= mid))]:
            for hold_label, col in [("跟踪止盈", "ret_trail"), ("持有1天", "ret_h1")]:
                print(fmt(f"{cfg_name}·{seg_label}·{hold_label}", stats(seg, col)))

    print(f"\n{'='*100}")
    print("消融表 · 跟踪止盈口径 (各条件单独叠加 BASE, 族内代表档)")
    print(f"{'='*100}")
    singles = {
        "C1 多头持续≥7日": pl.col("bull_align_days") >= 7,
        "C2 MA20斜率>0": pl.col("ma20_slope_5d") > 0,
        "C3 站上MA60且不向下": (pl.col("close") > pl.col("ma60")) & (pl.col("ma60_slope_5d") >= 0),
        "C4 低点抬高": pl.col("low_uplift_10d") > 0,
        "C5 近10日上涨≥6天": pl.col("up_days_10d") >= 6,
        "C6 回归R²≥0.65": (pl.col("reg_slope_20d") > 0) & (pl.col("reg_r2_20d") >= 0.65),
        "C7 ADX(10)≥30且+DI>-DI": (pl.col("adx_10") >= 30) & (pl.col("plus_di_10") > pl.col("minus_di_10")),
        "C8a 涨跌日量比≥1.2": pl.col("updown_vol_ratio_10d") >= 1.2,
        "C8b OBV20均线向上": pl.col("obv_ma20_slope_5d") > 0,
    }
    print(fmt("BASE", stats(trades, "ret_trail")))
    for name, cond in singles.items():
        print(fmt(name, stats(trades.filter(cond.fill_null(False)), "ret_trail")))

    print(f"\n{'='*100}")
    print("新评分相关性 (COMBINED池 · 跟踪止盈口径)")
    print(f"{'='*100}")
    combined = trades.filter(_configs()["COMBINED(族内最优档×6)"].fill_null(False))
    score_correlation(combined, "ret_trail")

    print("\n完成。")


if __name__ == "__main__":
    main()
