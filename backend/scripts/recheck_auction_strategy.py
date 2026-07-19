#!/usr/bin/env python
"""用新框架复检 strong_momentum_auction (2026-07-19)。

原回测口径 (策略文件头): 42个交易日, 不含费用, T-1收盘选趋势票 → T开盘按竞价确认
→ 开盘价买 → T+1开盘价卖, 黄金区[1.2%,3%)边界为同段样本标定(样本内)。

本复检升级为 ablation_trend_upgrade 同款口径:
  - 分钟K覆盖 2026-03-27 ~ 2026-07-17 (76天), 比原42天多出5月+7月中旬 → 部分样本外
  - 竞价量 = T日 09:30 分钟K成交量 (与生产 auction_strength_service 09:32校正路径同源);
    竞价量比 = 竞价量 / T-1全天量 (单位一致, 实测无需换算)
  - 可执行价: T开盘买 (T开盘涨幅>9.7%视为无法成交剔除), 出场对比:
    T+1开盘(现行定位) / T+2开盘 / 破MA10收盘离场(≤10日)
  - 全部净收益, 扣实际费率 ≈0.169% (佣金万0.85+印花税+过户+滑点)
  - 量比分档重验黄金区边界 + 前后分段 walk-forward
  - 附加: v2 趋势条件 (MA20斜率>0 + 回归R²≥0.60) 是否值得同步到本策略

复用 build_trades: 其「信号日didx → didx+1开盘买」即本策略「T-1选股 → T开盘买」,
ret_h1 = T+1开盘卖(现行定位), gap_t1 = T日开盘涨幅。

用法 (backend/): .venv/Scripts/python.exe -m scripts.recheck_auction_strategy
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import polars as pl

from scripts.ablation_trend_upgrade import (
    DATA_DIR,
    _add_adx_10,
    _load_panel,
    build_trades,
    fmt,
    stats,
)

logger = logging.getLogger(__name__)

GOLDEN_MIN, GOLDEN_MAX = 0.012, 0.03
DISTRIB_GAP, DISTRIB_RATIO = 0.05, 0.05

RATIO_BUCKETS = [
    ("<0.2% 极致缩量", 0.0, 0.002),
    ("0.2%~0.4% 缩量", 0.002, 0.004),
    ("0.4%~0.7% 标准量", 0.004, 0.007),
    ("0.7%~1.2% 放量下沿", 0.007, 0.012),
    ("1.2%~3% 黄金区", 0.012, 0.03),
    ("3%~5% 放量上沿", 0.03, 0.05),
    (">=5% 巨量", 0.05, 9.9),
]


def _load_auction_vols() -> pl.DataFrame:
    """全部分钟K日期的 09:30 竞价K线量: (auction_date, symbol, auction_vol)。"""
    frames = []
    for p in sorted((DATA_DIR / "kline_minute").glob("date=*")):
        d = p.name.split("=")[1]
        fp = p / "part.parquet"
        if not fp.exists():
            continue
        bars = (
            pl.scan_parquet(str(fp))
            .filter(pl.col("datetime").dt.time() == dt.time(1, 30))   # 09:30 BJ = 01:30 UTC
            .select(
                pl.lit(d).alias("auction_date"),
                "symbol",
                pl.col("volume").alias("auction_vol"),
            )
            .collect()
        )
        frames.append(bars)
    out = pl.concat(frames)
    logger.info("竞价K线: %d 天, %d 行", out["auction_date"].n_unique(), out.height)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None,
                    help="只统计该日期(含)之后的买入日, 如 2026-07-20 (封版后前瞻段复评用)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    panel = _load_panel()
    panel = _add_adx_10(panel)
    trades = build_trades(panel)   # BASE = auction 版趋势层 (同一四条规则+大盘池)

    # T日(买入日) 竞价量比 join: buy didx = didx+1 → buy_date
    days_df = panel.select(pl.col("date")).unique().sort("date").with_row_index("didx")
    buy_dates = days_df.with_columns(
        (pl.col("didx") - 1).alias("didx"),
        pl.col("date").cast(pl.Utf8).alias("buy_date"),
    ).select(["didx", "buy_date"])
    trades = trades.join(buy_dates, on="didx", how="left")

    auction = _load_auction_vols()
    trades = trades.join(
        auction.rename({"auction_date": "buy_date"}), on=["symbol", "buy_date"], how="left",
    ).with_columns(
        pl.when(pl.col("volume") > 0)
          .then(pl.col("auction_vol") / pl.col("volume"))   # volume=信号日(T-1)全天量
          .otherwise(None).alias("auc_ratio"),
    )

    # 只保留买入日有分钟K覆盖的信号日
    covered = trades.filter(pl.col("auc_ratio").is_not_null())
    if args.since:
        covered = covered.filter(pl.col("buy_date") >= args.since)
    n_days = covered["didx"].n_unique()
    print(f"\n{'='*102}")
    print(f"strong_momentum_auction 复检: 买入日分钟K覆盖 {n_days} 个信号日 "
          f"({covered['buy_date'].min()} ~ {covered['buy_date'].max()}), 净收益已扣费0.169%")
    print(f"{'='*102}\n")

    golden = (pl.col("auc_ratio") >= GOLDEN_MIN) & (pl.col("auc_ratio") < GOLDEN_MAX)
    distrib = (pl.col("gap_t1") > DISTRIB_GAP) & (pl.col("auc_ratio") >= DISTRIB_RATIO)
    v2_trend = (pl.col("ma20_slope_5d") > 0) & (pl.col("reg_slope_20d") > 0) & (pl.col("reg_r2_20d") >= 0.60)

    configs = {
        "纯趋势层(无竞价过滤)": pl.lit(True),
        "现行策略(黄金区+排除出货)": golden & ~distrib,
        "趋势层+v2两条(无竞价)": v2_trend,
        "现行策略+v2两条": golden & ~distrib & v2_trend,
    }

    print("── 配置 × 出场口径 (T开盘买入) ──")
    for name, mask in configs.items():
        sub = covered.filter(mask.fill_null(False))
        print(f"\n  [{name}]")
        for label, col in [("T+1开盘卖(现行)", "ret_h1"), ("T+3开盘卖(持有3日)", "ret_h3"),
                           ("破MA10收盘离场(≤10日)", "ret_trail")]:
            print("  " + fmt(label, stats(sub, col)))

    print(f"\n{'='*102}")
    print("竞价量比分档重验 (纯趋势层内分档, T+1开盘卖) — 检验黄金区边界是否复现")
    print(f"{'='*102}")
    for label, lo, hi in RATIO_BUCKETS:
        sub = covered.filter((pl.col("auc_ratio") >= lo) & (pl.col("auc_ratio") < hi))
        print(fmt(label, stats(sub, "ret_h1")))

    print(f"\n{'='*102}")
    print("walk-forward 前后分段 (现行策略配置)")
    print(f"{'='*102}")
    didx_list = sorted(covered["didx"].unique().to_list())
    mid = didx_list[len(didx_list) // 2]
    cur = covered.filter((golden & ~distrib).fill_null(False))
    for seg_label, seg in [("前半", cur.filter(pl.col("didx") < mid)),
                           ("后半", cur.filter(pl.col("didx") >= mid))]:
        rng = (seg["buy_date"].min(), seg["buy_date"].max()) if seg.height else ("-", "-")
        for hold_label, col in [("T+1开盘卖", "ret_h1"), ("破MA10离场", "ret_trail")]:
            print(fmt(f"现行·{seg_label}({rng[0]}~{rng[1]})·{hold_label}", stats(seg, col)))

    # ── 候选数/市况开关 (2026-07-19 复检清单延伸, 网格预定一次跑完) ──
    # 门控信号 = T-1收盘纯趋势层候选数 (决策时点可知, 无未来函数)。
    # 逻辑: 趋势层候选多 = 市场有普涨动量的市况; 候选稀少 = 逆势硬做。
    print(f"\n{'='*102}")
    print("候选数/市况开关 (现行策略, T+1开盘卖) — 门控: T-1收盘趋势层候选数")
    print(f"{'='*102}")
    trend_count = trades.group_by("didx").agg(pl.len().alias("trend_n"))
    cur2 = cur.join(trend_count, on="didx", how="left")
    q1, q2 = trend_count["trend_n"].quantile(0.33), trend_count["trend_n"].quantile(0.67)
    print(f"趋势层候选数分布: 三分位界 {q1:.0f}/{q2:.0f}, "
          f"中位 {trend_count['trend_n'].median():.0f}, 均值 {trend_count['trend_n'].mean():.1f}")
    for label, m in [(f"低候选段(<{q1:.0f})", pl.col("trend_n") < q1),
                     (f"中候选段([{q1:.0f},{q2:.0f}))", (pl.col("trend_n") >= q1) & (pl.col("trend_n") < q2)),
                     (f"高候选段(≥{q2:.0f})", pl.col("trend_n") >= q2)]:
        print(fmt(label, stats(cur2.filter(m), "ret_h1")))
    for th in (10, 20, 30, 40):
        print(fmt(f"开关: 仅趋势候选≥{th}日参与", stats(cur2.filter(pl.col("trend_n") >= th), "ret_h1")))

    # 原口径对照: 不扣费的现行策略 T+1开盘卖 (与策略文件头 +0.83%/53.3% 直接可比)
    print(f"\n{'='*102}")
    print("原口径对照 (现行策略, T+1开盘卖, 不扣费 — 与文件头 笔均+0.83%/胜率53.3% 可比)")
    print(f"{'='*102}")
    from scripts.ablation_trend_upgrade import FEE_ROUND_TRIP
    gross = cur.with_columns((pl.col("ret_h1") + FEE_ROUND_TRIP).alias("ret_gross"))
    print(fmt("现行策略(不扣费)", stats(gross, "ret_gross")))

    print("\n完成。")


if __name__ == "__main__":
    main()
