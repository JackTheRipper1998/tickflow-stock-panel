#!/usr/bin/env python
"""ai_20260627「短期动量 → 明确上涨趋势」升级的消融回测 (2026-07-19)。

设计 (与需求文档逐条对应):
  - 可执行价口径: T日收盘选股 → T+1 开盘价买入 → 按持有口径以 开盘/收盘 价卖出。
    禁止 "T+2 最高价" 之类不可执行口径。T+1 开盘涨幅 >9.7% 视为一字/准一字板, 无法
    买入, 剔除该笔。
  - 扣费: 每笔往返扣 0.25% (印花税+佣金+滑点)。
  - 新市值池: 总市值≥200亿 + 流通市值≥100亿 (instruments 最新快照近似历史股本),
    沪深主板, 非ST, 价格3~300, 成交额≥0.8亿。
  - 消融: BASE = 新池 + 原四条形态规则; 每个新条件单独叠加在 BASE 上对比,
    参数网格预先定死一次跑完, 不看结果回调。
  - 保留标准 (预定): 笔均净收益 高于 BASE 且 日均候选 ≥ 8 只。
  - 组合配置跑: 持有期敏感性 (1天/3天/5天/破MA10跟踪止盈≤10天) + 前后分段
    walk-forward + 新评分与前瞻收益相关性检验。

用法 (backend/ 目录):
    .venv/Scripts/python.exe -m scripts.ablation_trend_upgrade
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

FEE_ROUND_TRIP = 0.0025          # 往返费用
LIMIT_OPEN_SKIP = 0.097          # T+1 开盘涨幅超过该值视为无法买入
MIN_AVG_CANDIDATES = 8.0         # 条件保留标准: 日均候选下限
MAX_TRAIL_DAYS = 10              # 跟踪止盈最长持有

# 原四条形态规则默认参数 (与 ai_20260627 一致)
MOMENTUM_5D_MIN = 0.05
MOMENTUM_10D_MIN = 0.08
NEAR_HIGH_PCT = 0.10
VOL_RATIO_MIN = 0.70


def _load_panel() -> pl.DataFrame:
    from app.indicators.pipeline import compute_all

    enriched = pl.scan_parquet(str(DATA_DIR / "kline_daily_enriched" / "**" / "*.parquet")).collect()
    inst = pl.read_parquet(str(DATA_DIR / "instruments" / "instruments.parquet"))
    panel = compute_all(enriched, instruments=inst)
    join_cols = [c for c in ["symbol", "name", "total_shares", "float_shares"] if c in inst.columns]
    panel = panel.join(inst.select(join_cols).unique(subset=["symbol"]), on="symbol", how="left")
    return panel.sort(["symbol", "date"])


def _add_adx_10(panel: pl.DataFrame) -> pl.DataFrame:
    """脚本内补算 ADX 周期10 变体 (管道只带默认周期14)。"""
    prev_close = pl.col("close").shift(1).over("symbol")
    up_move = pl.col("high") - pl.col("high").shift(1).over("symbol")
    down_move = pl.col("low").shift(1).over("symbol") - pl.col("low")
    panel = panel.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        ).alias("_tr"),
        pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0).alias("_dmp"),
        pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0).alias("_dmm"),
    )
    a = 1.0 / 10
    panel = panel.with_columns(
        pl.col("_tr").ewm_mean(alpha=a, adjust=False).over("symbol").alias("_tr_s"),
        pl.col("_dmp").ewm_mean(alpha=a, adjust=False).over("symbol").alias("_dmp_s"),
        pl.col("_dmm").ewm_mean(alpha=a, adjust=False).over("symbol").alias("_dmm_s"),
    ).with_columns(
        pl.when(pl.col("_tr_s") > 0).then(100 * pl.col("_dmp_s") / pl.col("_tr_s")).otherwise(None).alias("plus_di_10"),
        pl.when(pl.col("_tr_s") > 0).then(100 * pl.col("_dmm_s") / pl.col("_tr_s")).otherwise(None).alias("minus_di_10"),
    )
    disum = pl.col("plus_di_10") + pl.col("minus_di_10")
    panel = panel.with_columns(
        pl.when(disum > 0)
          .then(100 * (pl.col("plus_di_10") - pl.col("minus_di_10")).abs() / disum)
          .otherwise(None).alias("_dx10"),
    ).with_columns(
        pl.col("_dx10").ewm_mean(alpha=a, adjust=False).over("symbol").alias("adx_10"),
    )
    return panel.drop(["_tr", "_dmp", "_dmm", "_tr_s", "_dmp_s", "_dmm_s", "_dx10"])


def _base_mask() -> pl.Expr:
    """新市值池 + 原四条形态规则 (BASE)。"""
    return (
        (pl.col("close") >= 3) & (pl.col("close") <= 300)
        & (pl.col("amount") >= 0.8e8)
        & (pl.col("close") * pl.col("total_shares") >= 20e9)
        & (pl.col("close") * pl.col("float_shares") >= 10e9)
        & ~pl.col("name").str.contains("(?i)ST|\\*ST|退")
        & ~pl.col("symbol").str.starts_with("688")
        & ~pl.col("symbol").str.starts_with("300")
        & ~pl.col("symbol").str.starts_with("301")
        & ~pl.col("symbol").str.ends_with(".BJ")
        # 原四条形态规则
        & (pl.col("ma5") > pl.col("ma10")) & (pl.col("ma10") > pl.col("ma20"))
        & (pl.col("close") > pl.col("ma20"))
        & (pl.col("momentum_5d") >= MOMENTUM_5D_MIN)
        & (pl.col("momentum_10d") >= MOMENTUM_10D_MIN)
        & (pl.col("close") >= pl.col("high_60d") * (1.0 - NEAR_HIGH_PCT))
        & (pl.col("vol_ratio_5d") >= VOL_RATIO_MIN)
    )


def build_trades(panel: pl.DataFrame) -> pl.DataFrame:
    """构建候选交易大表: BASE 命中 × 可执行入场 × 各持有口径收益。

    返回每行一笔潜在交易, 含全部趋势质量指标列 (供消融过滤) 和:
      ret_h1/ret_h3/ret_h5 (T+1开盘买 → k个交易日后开盘卖, 净收益)
      ret_trail (破MA10收盘离场, ≤10日, 净收益)
    """
    days_df = (
        panel.select(pl.col("date")).unique().sort("date")
        .with_row_index("didx")
    )
    panel = panel.join(days_df, on="date", how="left")

    # (symbol, didx) → open/close/ma10 查找表
    lookup = panel.select(["symbol", "didx", "open", "close", "ma10"])

    cands = panel.filter(_base_mask().fill_null(False))
    logger.info("BASE 命中: %d 行, 信号日 %d 个", cands.height, cands["date"].n_unique())

    # 入场: T+1 开盘
    def _join_price(df: pl.DataFrame, offset: int, cols: dict[str, str]) -> pl.DataFrame:
        sel = lookup.rename({v: k for k, v in cols.items()}).select(["symbol", "didx", *cols.keys()])
        return df.join(
            sel.with_columns((pl.col("didx") - offset).alias("didx")),
            on=["symbol", "didx"], how="left",
        )

    cands = _join_price(cands, 1, {"buy_open": "open"})
    cands = _join_price(cands, 2, {"sell_open_h1": "open"})
    cands = _join_price(cands, 4, {"sell_open_h3": "open"})
    cands = _join_price(cands, 6, {"sell_open_h5": "open"})

    # 可执行性: T+1 有行情 (未停牌) 且非一字/准一字板开盘
    cands = cands.with_columns(
        (pl.col("buy_open") / pl.col("close") - 1).alias("gap_t1"),
    ).with_columns(
        (pl.col("buy_open").is_not_null() & (pl.col("buy_open") > 0)
         & (pl.col("gap_t1") <= LIMIT_OPEN_SKIP)).alias("fillable"),
    )

    for k, col in [(1, "sell_open_h1"), (3, "sell_open_h3"), (5, "sell_open_h5")]:
        cands = cands.with_columns(
            pl.when(pl.col("fillable") & pl.col(col).is_not_null() & (pl.col(col) > 0))
              .then(pl.col(col) / pl.col("buy_open") - 1 - FEE_ROUND_TRIP)
              .otherwise(None).alias(f"ret_h{k}"),
        )

    # 跟踪止盈: 买入日起, 首个 close<ma10 的收盘离场; 10日无触发按第10日收盘离场
    trail = _compute_trailing(cands, lookup)
    cands = cands.join(trail, on=["symbol", "didx"], how="left")
    return cands


def _compute_trailing(cands: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    """逐笔计算破MA10跟踪止盈收益 (python 循环, 样本量在几千笔级, 可接受)。"""
    import numpy as np

    by_symbol: dict[str, dict] = {}
    for sym, sub in lookup.group_by("symbol"):
        s = sub.sort("didx")
        by_symbol[sym[0]] = {
            "didx": s["didx"].to_numpy(),
            "close": s["close"].to_numpy(),
            "ma10": s["ma10"].to_numpy(),
        }

    rows = []
    for r in cands.select(["symbol", "didx", "buy_open", "fillable"]).iter_rows(named=True):
        if not r["fillable"] or r["buy_open"] is None:
            rows.append({"symbol": r["symbol"], "didx": r["didx"], "ret_trail": None, "trail_days": None})
            continue
        arr = by_symbol.get(r["symbol"])
        if arr is None:
            rows.append({"symbol": r["symbol"], "didx": r["didx"], "ret_trail": None, "trail_days": None})
            continue
        pos = np.searchsorted(arr["didx"], r["didx"] + 1)   # 买入日行
        exit_price = None
        days_held = None
        end = min(pos + MAX_TRAIL_DAYS, len(arr["didx"]))
        for j in range(pos, end):
            c, m = arr["close"][j], arr["ma10"][j]
            if c is not None and m is not None and not (np.isnan(c) or np.isnan(m)) and c < m:
                exit_price, days_held = c, j - pos + 1
                break
        if exit_price is None and end > pos:
            exit_price, days_held = arr["close"][end - 1], end - pos
        ret = (float(exit_price) / r["buy_open"] - 1 - FEE_ROUND_TRIP) if exit_price is not None else None
        rows.append({"symbol": r["symbol"], "didx": r["didx"], "ret_trail": ret,
                     "trail_days": days_held})
    return pl.DataFrame(rows, schema={"symbol": pl.Utf8, "didx": pl.UInt32,
                                      "ret_trail": pl.Float64, "trail_days": pl.Int64})


def stats(trades: pl.DataFrame, ret_col: str = "ret_h1") -> dict:
    """一组交易的指标: 信号日数/日均候选/笔数/笔均/胜率/日组合累计/最差单日。"""
    n_days = trades["didx"].n_unique()
    if n_days == 0:
        return {"days": 0, "avg_cands": 0.0, "n": 0, "mean": None, "win": None,
                "cum": None, "worst_day": None}
    avg_cands = trades.height / n_days
    t = trades.filter(pl.col(ret_col).is_not_null())
    if t.is_empty():
        return {"days": n_days, "avg_cands": avg_cands, "n": 0, "mean": None,
                "win": None, "cum": None, "worst_day": None}
    daily = t.group_by("didx").agg(pl.col(ret_col).mean().alias("day_ret"))
    return {
        "days": n_days,
        "avg_cands": avg_cands,
        "n": t.height,
        "mean": t[ret_col].mean(),
        "win": (t[ret_col] > 0).sum() / t.height,
        "cum": daily["day_ret"].sum(),
        "worst_day": daily["day_ret"].min(),
    }


def fmt(name: str, s: dict) -> str:
    if s["n"] == 0:
        return f"{name:<34} 日均{s['avg_cands']:>6.1f}  无有效笔"
    return (f"{name:<34} 日均{s['avg_cands']:>6.1f}  笔数{s['n']:>5}  "
            f"笔均{s['mean']*100:>+6.2f}%  胜率{s['win']*100:>5.1f}%  "
            f"日组合累计{s['cum']*100:>+7.1f}%  最差日{s['worst_day']*100:>+6.2f}%")


# ── 消融条件网格 (预先定死) ──────────────────────────────────────────
def ablation_grid() -> list[tuple[str, pl.Expr]]:
    grid: list[tuple[str, pl.Expr]] = []
    for d in (5, 6, 7, 8, 10):
        grid.append((f"C1 多头持续≥{d}日", pl.col("bull_align_days") >= d))
    grid.append(("C2 MA20斜率>0", pl.col("ma20_slope_5d") > 0))
    grid.append(("C3 站上MA60且MA60不向下", (pl.col("close") > pl.col("ma60")) & (pl.col("ma60_slope_5d") >= 0)))
    grid.append(("C4 低点抬高", pl.col("low_uplift_10d") > 0))
    grid.append(("C5 近10日上涨≥6天", pl.col("up_days_10d") >= 6))
    for r2 in (0.60, 0.65, 0.70):
        grid.append((f"C6 回归斜率>0且R²≥{r2:.2f}",
                     (pl.col("reg_slope_20d") > 0) & (pl.col("reg_r2_20d") >= r2)))
    for period in (14, 10):
        for th in (20, 25, 30):
            grid.append((f"C7 ADX({period})≥{th}且+DI>-DI",
                         (pl.col(f"adx_{period}") >= th)
                         & (pl.col(f"plus_di_{period}") > pl.col(f"minus_di_{period}"))))
    grid.append(("C8a 涨跌日量比≥1.2", pl.col("updown_vol_ratio_10d") >= 1.2))
    grid.append(("C8b OBV20均线向上", pl.col("obv_ma20_slope_5d") > 0))
    return grid


def score_correlation(trades: pl.DataFrame, ret_col: str) -> None:
    """新评分(趋势质量导向)与前瞻收益的相关性 (逐信号日截面归一化)。"""
    t = trades.filter(pl.col(ret_col).is_not_null() & pl.col("reg_slope_20d").is_not_null()
                      & pl.col("reg_r2_20d").is_not_null() & pl.col("adx_14").is_not_null())
    if t.height < 30:
        print("  样本不足, 跳过评分相关性检验")
        return
    t = t.with_columns((pl.col("reg_slope_20d") * 250 * pl.col("reg_r2_20d")).alias("_slope_q"))

    def _norm(col: str) -> pl.Expr:
        lo, hi = pl.col(col).min().over("didx"), pl.col(col).max().over("didx")
        return pl.when(hi > lo).then((pl.col(col) - lo) / (hi - lo)).otherwise(0.5)

    t = t.with_columns(
        (
            _norm("_slope_q") * 0.4
            + _norm("bull_align_days") * 0.2
            + _norm("adx_14") * 0.2
            + _norm("updown_vol_ratio_10d").fill_null(0.5) * 0.2
        ).alias("_score"),
    ).with_columns(
        pl.col("_score").rank(descending=True).over("didx").alias("_rank"),
        pl.count().over("didx").alias("_total"),
    ).with_columns((pl.col("_rank") / pl.col("_total")).alias("_rank_pct"))

    pearson = t.select(pl.corr("_score", ret_col)).item()
    spearman = t.select(pl.corr("_rank_pct", ret_col, method="spearman")).item()
    print(f"  新评分 vs {ret_col}  (n={t.height}):  Pearson(评分,收益)={pearson:+.3f}  "
          f"Spearman(排名百分位,收益)={spearman:+.3f}")
    print("  (Spearman为负=排名越靠前收益越高=评分有效; |corr|<0.1 视为噪声)")
    buckets = t.with_columns((pl.col("_rank_pct") * 5).ceil().clip(1, 5).cast(pl.Int32).alias("_b"))
    for b in range(1, 6):
        sub = buckets.filter(pl.col("_b") == b)
        if sub.is_empty():
            continue
        print(f"    第{b}组(评分{'最靠前' if b == 1 else f'第{b}档'}20%): n={sub.height:>5} "
              f"笔均{sub[ret_col].mean()*100:+.2f}% 胜率{(sub[ret_col] > 0).sum()/sub.height*100:.1f}%")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    panel = _load_panel()
    panel = _add_adx_10(panel)
    trades = build_trades(panel)

    date_min, date_max = trades["date"].min(), trades["date"].max()
    print(f"\n{'='*100}")
    print(f"消融回测: T收盘选股 → T+1开盘买入(开盘涨幅>{LIMIT_OPEN_SKIP*100:.1f}%剔除) → 可执行价卖出, "
          f"扣费{FEE_ROUND_TRIP*100:.2f}%")
    print(f"窗口: {date_min} ~ {date_max}   池: 总市值≥200亿+流通≥100亿 主板非ST")
    print(f"{'='*100}\n")

    base_s = stats(trades)
    print("── 第一步: BASE (新池 + 原四条形态规则), 持有1天 (T+2开盘卖) ──")
    print(fmt("BASE", base_s))

    print("\n── 第二步: 逐条叠加消融 (每条单独加在 BASE 上, 持有1天) ──")
    kept: list[tuple[str, pl.Expr]] = []
    for name, cond in ablation_grid():
        sub = trades.filter(cond.fill_null(False))
        s = stats(sub)
        mark = ""
        if s["n"] > 0 and s["mean"] is not None and base_s["mean"] is not None:
            improved = s["mean"] > base_s["mean"]
            enough = s["avg_cands"] >= MIN_AVG_CANDIDATES
            if improved and enough:
                mark = "  ← 保留"
                kept.append((name, cond))
            elif improved:
                mark = "  (笔均↑但池子过小)"
        print(fmt(name, s) + mark)

    print(f"\n保留条件 ({len(kept)}): {[n for n, _ in kept] or '无 — 全部未过标准'}")

    # ── 第三步: 组合配置 ──
    combined_mask = pl.lit(True)
    for _, cond in kept:
        combined_mask = combined_mask & cond.fill_null(False)
    combined = trades.filter(combined_mask)
    print("\n── 第三步: 组合配置 (BASE + 全部保留条件) 持有期敏感性 ──")
    for label, col in [("持有1天(T+2开盘卖)", "ret_h1"), ("持有3天(T+4开盘卖)", "ret_h3"),
                       ("持有5天(T+6开盘卖)", "ret_h5"), ("破MA10收盘离场(≤10日)", "ret_trail")]:
        print(fmt(f"组合 · {label}", stats(combined, col)))
    if "trail_days" in combined.columns and combined.height:
        td = combined.filter(pl.col("trail_days").is_not_null())["trail_days"]
        if td.len():
            print(f"    跟踪止盈平均持有 {td.mean():.1f} 天, 中位 {td.median():.0f} 天")

    # ── 第四步: 前后分段 walk-forward ──
    print("\n── 第四步: 前后分段 walk-forward (组合配置, 持有1天) ──")
    didx_list = sorted(trades["didx"].unique().to_list())
    mid = didx_list[len(didx_list) // 2]
    for label, seg in [("前半段", combined.filter(pl.col("didx") < mid)),
                       ("后半段", combined.filter(pl.col("didx") >= mid))]:
        seg_dates = (seg["date"].min(), seg["date"].max()) if seg.height else ("-", "-")
        print(fmt(f"组合·{label} {seg_dates[0]}~{seg_dates[1]}", stats(seg)))
        print(fmt(f"BASE·{label}", stats(
            trades.filter((pl.col("didx") < mid) if label == "前半段" else (pl.col("didx") >= mid)))))

    # ── 第五步: 新评分相关性检验 ──
    print("\n── 第五步: 新评分(趋势质量导向)与前瞻收益相关性 (组合池, 持有1天) ──")
    score_correlation(combined, "ret_h1")

    print("\n完成。")


if __name__ == "__main__":
    main()
