#!/usr/bin/env python
"""分析某策略在分钟K覆盖窗口内, 逐个信号日选出「T+1 日中买入 -> T+2 最高价卖出」
收益最高的候选股, 拼成一串每日一只的股票链, 并输出该链的明细与整体表现。

用法（从 backend/ 目录运行）：
    .venv/bin/python -m scripts.analyze_candidate_paths --strategy ai_20260627
    .venv/bin/python -m scripts.analyze_candidate_paths --strategy ai_20260627 --entry-time 13:00
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
STRATEGY_DIRS = [
    Path(__file__).resolve().parent.parent / "app" / "strategy" / "builtin",
    DATA_DIR / "strategies" / "custom",
    DATA_DIR / "strategies" / "ai",
]

def _find_strategy_file(strategy_id: str) -> Path:
    for d in STRATEGY_DIRS:
        p = d / f"{strategy_id}.py"
        if p.exists():
            return p
    raise FileNotFoundError(f"策略文件未找到: {strategy_id}.py")


def _load_daily_panel() -> pl.DataFrame:
    from app.indicators.pipeline import compute_all

    enriched = pl.scan_parquet(str(DATA_DIR / "kline_daily_enriched" / "**" / "*.parquet")).collect()
    inst = pl.read_parquet(str(DATA_DIR / "instruments" / "instruments.parquet"))
    enriched = compute_all(enriched, instruments=inst)
    return enriched.join(
        inst.select(["symbol", "name"]).unique(subset=["symbol"]), on="symbol", how="left"
    )


def _minute_trading_days() -> list[str]:
    return sorted(p.name.split("=")[-1] for p in (DATA_DIR / "kline_minute").glob("date=*"))


def _score_and_rank(hits: pl.DataFrame, scoring: dict[str, float], limit: int) -> pl.DataFrame:
    if not scoring or hits.is_empty():
        return hits.head(limit).with_row_index("rank", offset=1)
    score_expr = None
    for col, w in scoring.items():
        if col not in hits.columns:
            continue
        vmin, vmax = hits[col].min(), hits[col].max()
        vrange = (vmax - vmin) if (vmax is not None and vmin is not None) else 0
        normalized = ((pl.col(col) - vmin) / vrange) if vrange else pl.lit(0.5)
        part = normalized * (w / sum(scoring.values()))
        score_expr = part if score_expr is None else score_expr + part
    if score_expr is None:
        return hits.head(limit).with_row_index("rank", offset=1)
    ranked = hits.with_columns((score_expr * 100).alias("_score")).sort("_score", descending=True)
    total = ranked.height
    return ranked.head(limit).with_row_index("rank", offset=1).with_columns(pl.lit(total).alias("_total"))


def _bj_to_utc(d: str, hh_mm: str) -> datetime:
    h, m = map(int, hh_mm.split(":"))
    return datetime.strptime(d, "%Y-%m-%d").replace(hour=(h - 8) % 24, minute=m)


def _entry_price_at(minute_cache: dict, sym: str, day: str, entry_time: str) -> float | None:
    if day not in minute_cache:
        fp = DATA_DIR / "kline_minute" / f"date={day}" / "part.parquet"
        minute_cache[day] = pl.read_parquet(str(fp)).sort("datetime") if fp.exists() else pl.DataFrame()
    mdf = minute_cache[day]
    if mdf.is_empty():
        return None
    sub = mdf.filter(pl.col("symbol").str.starts_with(sym))
    if sub.is_empty():
        return None
    target_dt = _bj_to_utc(day, entry_time)
    s = sub.filter(pl.col("datetime") <= target_dt)
    if s.is_empty():
        return None
    v = s["close"][-1]
    return float(v) if v is not None else None


def run(strategy_id: str, hold_days: int, entry_time: str) -> None:
    from app.strategy.engine import StrategyEngine

    strategy_path = _find_strategy_file(strategy_id)
    strategy = StrategyEngine._load_file(strategy_path)
    scoring = strategy.meta.get("scoring", {})
    limit = int(strategy.meta.get("limit", 50))
    logger.info("策略: %s (%s)  limit=%d  入场时点=%s", strategy.meta.get("name"), strategy_id, limit, entry_time)

    enriched = _load_daily_panel()
    all_days = sorted(enriched.select("date").unique().to_series().cast(pl.Utf8).to_list())
    minute_days = _minute_trading_days()

    basic_expr = StrategyEngine._basic_filter_expr(enriched, strategy.basic_filter)
    daily_lookup = enriched.select(["symbol", "date", "high", "name"]).with_columns(
        pl.col("date").cast(pl.Utf8)
    )

    minute_cache: dict[str, pl.DataFrame] = {}
    trades = []
    all_trades = []
    for d in minute_days:
        if d not in all_days:
            continue
        di = all_days.index(d)
        if di + hold_days + 1 >= len(all_days):
            continue
        buy_date = all_days[di + 1]
        sell_date = all_days[di + 1 + hold_days]
        if buy_date not in minute_days:
            continue

        day_df = enriched.filter(pl.col("date") == datetime.strptime(d, "%Y-%m-%d").date())
        if day_df.is_empty():
            continue
        mask = day_df.select(
            ((basic_expr if basic_expr is not None else pl.lit(True)) & strategy.filter_fn(day_df, {})).alias("_hit")
        )["_hit"].fill_null(False)
        hits = day_df.filter(mask)
        if hits.is_empty():
            continue
        ranked = _score_and_rank(hits, scoring, limit)
        total = ranked["_total"][0] if "_total" in ranked.columns and ranked.height else ranked.height

        day_trades = []
        for row in ranked.iter_rows(named=True):
            sym = row["symbol"]
            entry_price = _entry_price_at(minute_cache, sym, buy_date, entry_time)
            if entry_price is None or entry_price <= 0:
                continue
            sell_row = daily_lookup.filter((pl.col("symbol") == sym) & (pl.col("date") == sell_date))
            if sell_row.is_empty():
                continue
            exit_price = sell_row["high"][0]
            if exit_price is None:
                continue
            ret = (exit_price - entry_price) / entry_price
            day_trades.append({
                "symbol": sym,
                "name": row.get("name") or "",
                "signal_date": d,
                "buy_date": buy_date,
                "sell_date": sell_date,
                "rank": row["rank"],
                "total": total,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "ret": float(ret),
            })

        if not day_trades:
            continue
        all_trades.extend(day_trades)
        # 每天只选当日候选中收益最高的一只, 拼成一串
        best_of_day = max(day_trades, key=lambda t: t["ret"])
        trades.append(best_of_day)

    if not trades:
        logger.warning("无有效样本")
        return

    df = pl.DataFrame(trades).sort("signal_date")
    compounded = 1.0
    for r in df["ret"].to_list():
        compounded *= (1 + r)
    compounded -= 1.0

    print(f"\n入场时点: T+1 {entry_time}  出场: T+{1+hold_days} 最高价")
    print(f"链条长度: {df.height} 天  (信号日: {df['signal_date'].to_list()[0]} ~ {df['signal_date'].to_list()[-1]})")
    print(f"逐日平均收益: {df['ret'].mean()*100:.2f}%  中位收益: {df['ret'].median()*100:.2f}%  "
          f"胜率: {(df['ret'] > 0).sum() / df.height * 100:.1f}%")
    print(f"若逐日满仓复利串联, 累计收益: {compounded*100:+.2f}%\n")

    print(f"{'信号日':<12}{'买入日':<12}{'代码':<10}{'名称':<8}{'排名':<8}"
          f"{'入场价':>8}{'出场价':>8}{'收益':>9}")
    for rec in df.iter_rows(named=True):
        print(f"{rec['signal_date']:<12}{rec['buy_date']:<12}{rec['symbol']:<10}{rec['name']:<8}"
              f"{rec['rank']}/{rec['total']:<6}{rec['entry_price']:>8.2f}{rec['exit_price']:>8.2f}"
              f"{rec['ret']*100:>8.2f}%")

    _print_rank_return_correlation(all_trades)


def _print_rank_return_correlation(all_trades: list[dict]) -> None:
    adf = pl.DataFrame(all_trades).with_columns((pl.col("rank") / pl.col("total")).alias("rank_pct"))
    pearson_rank = adf.select(pl.corr("rank", "ret")).item()
    pearson_pct = adf.select(pl.corr("rank_pct", "ret")).item()
    spearman = adf.select(pl.corr("rank_pct", "ret", method="spearman")).item()

    print(f"\n评分排名 vs 实际收益 相关性  (全部候选样本 n={adf.height})")
    print(f"  Pearson(排名序号, 收益)     = {pearson_rank:+.3f}")
    print(f"  Pearson(排名百分位, 收益)   = {pearson_pct:+.3f}")
    print(f"  Spearman(排名百分位, 收益)  = {spearman:+.3f}")
    print("  (排名序号越小=评分越靠前; 相关系数为负说明评分越靠前收益越高, 即评分有效)")

    buckets = adf.with_columns(
        (pl.col("rank_pct") * 5).floor().clip(0, 4).cast(pl.Int32).alias("_bucket")
    )
    print(f"\n  按评分百分位分5组 (第1组=评分最靠前20%):")
    print(f"  {'分组':<10}{'样本数':>8}{'平均收益':>10}{'胜率':>8}{'亏损占比':>10}{'平均亏损':>10}")
    for b in range(5):
        sub = buckets.filter(pl.col("_bucket") == b)
        if sub.is_empty():
            continue
        losers = sub.filter(pl.col("ret") < 0)
        loss_rate = losers.height / sub.height * 100
        avg_loss = losers["ret"].mean() * 100 if losers.height else 0.0
        print(f"  第{b+1}组{'':<7}{sub.height:>8}{sub['ret'].mean()*100:>9.2f}%"
              f"{(sub['ret'] > 0).sum() / sub.height * 100:>7.1f}%"
              f"{loss_rate:>9.1f}%{avg_loss:>9.2f}%")

    losers = adf.filter(pl.col("ret") < 0)
    print(f"\n评分排名 vs 亏损 相关性  (仅亏损样本 n={losers.height}, 占比 {losers.height/adf.height*100:.1f}%)")
    if losers.height >= 2:
        loss_pearson_pct = losers.select(pl.corr("rank_pct", "ret")).item()
        loss_spearman = losers.select(pl.corr("rank_pct", "ret", method="spearman")).item()
        print(f"  Pearson(排名百分位, 亏损幅度)  = {loss_pearson_pct:+.3f}")
        print(f"  Spearman(排名百分位, 亏损幅度) = {loss_spearman:+.3f}")
        print("  (负值说明评分越靠后, 亏得越多; 接近0说明评分对亏损幅度无区分能力)")
    else:
        print("  样本太少, 无法计算相关性")


def main() -> None:
    ap = argparse.ArgumentParser(description="逐日选出策略候选中收益最高的个股, 拼成一串股票链")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--hold-days", type=int, default=1, help="T+1买入后再持有几个交易日卖出, 默认1 (即T+2卖出)")
    ap.add_argument("--entry-time", default="13:00", help="T+1 买入的日内时点 (北京时间 HH:MM), 默认 13:00")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(args.strategy, args.hold_days, args.entry_time)


if __name__ == "__main__":
    main()
