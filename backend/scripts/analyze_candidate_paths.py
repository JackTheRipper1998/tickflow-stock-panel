#!/usr/bin/env python
"""分析某策略在分钟K覆盖窗口内的候选个股 T+1 买入->T+2 卖出的收益分布,
找出最高/最低收益的个股, 标注其信号日排名, 并输出分钟级价格路径。

用法（从 backend/ 目录运行）：
    .venv/bin/python -m scripts.analyze_candidate_paths --strategy ai_20260627
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

PATH_TIMES = ["09:35", "09:50", "10:30", "11:00", "13:00", "13:30", "14:00", "14:30", "14:55"]


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


def _minute_path(minute_cache: dict, sym: str, day: str) -> dict[str, float | None]:
    if day not in minute_cache:
        fp = DATA_DIR / "kline_minute" / f"date={day}" / "part.parquet"
        minute_cache[day] = pl.read_parquet(str(fp)).sort("datetime") if fp.exists() else pl.DataFrame()
    mdf = minute_cache[day]
    if mdf.is_empty():
        return {}
    sub = mdf.filter(pl.col("symbol").str.starts_with(sym))
    if sub.is_empty():
        return {}
    path: dict[str, float | None] = {}
    if sub["open"][0] is not None:
        path["open"] = float(sub["open"][0])
    for t in PATH_TIMES:
        target_dt = _bj_to_utc(day, t)
        s = sub.filter(pl.col("datetime") <= target_dt)
        path[t] = float(s["close"][-1]) if not s.is_empty() else None
    if sub["close"][-1] is not None:
        path["close"] = float(sub["close"][-1])
    return path


def _print_path(label: str, path: dict[str, float | None], entry_price: float) -> None:
    print(f"  {label}:")
    keys = ["open"] + PATH_TIMES + ["close"]
    for k in keys:
        v = path.get(k)
        if v is None:
            continue
        chg = (v - entry_price) / entry_price * 100
        print(f"    {k:<8} {v:>8.2f}  ({chg:+.2f}% vs 买入价)")


def run(strategy_id: str, hold_days: int) -> None:
    from app.strategy.engine import StrategyEngine

    strategy_path = _find_strategy_file(strategy_id)
    strategy = StrategyEngine._load_file(strategy_path)
    scoring = strategy.meta.get("scoring", {})
    limit = int(strategy.meta.get("limit", 50))
    logger.info("策略: %s (%s)  limit=%d", strategy.meta.get("name"), strategy_id, limit)

    enriched = _load_daily_panel()
    all_days = sorted(enriched.select("date").unique().to_series().cast(pl.Utf8).to_list())
    minute_days = _minute_trading_days()

    basic_expr = StrategyEngine._basic_filter_expr(enriched, strategy.basic_filter)
    daily_lookup = enriched.select(["symbol", "date", "open", "close", "name"]).with_columns(
        pl.col("date").cast(pl.Utf8)
    )

    trades = []
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

        for row in ranked.iter_rows(named=True):
            sym = row["symbol"]
            buy_row = daily_lookup.filter((pl.col("symbol") == sym) & (pl.col("date") == buy_date))
            sell_row = daily_lookup.filter((pl.col("symbol") == sym) & (pl.col("date") == sell_date))
            if buy_row.is_empty() or sell_row.is_empty():
                continue
            entry_price = buy_row["open"][0]
            exit_price = sell_row["close"][0]
            if entry_price is None or exit_price is None or entry_price <= 0:
                continue
            ret = (exit_price - entry_price) / entry_price
            trades.append({
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

    if not trades:
        logger.warning("无有效样本")
        return

    df = pl.DataFrame(trades)
    print(f"\n总样本数: {df.height}  (信号日: {sorted(set(df['signal_date'].to_list()))})")
    print(f"平均收益: {df['ret'].mean()*100:.2f}%  中位收益: {df['ret'].median()*100:.2f}%  "
          f"胜率: {(df['ret'] > 0).sum() / df.height * 100:.1f}%")

    best = df.sort("ret", descending=True).row(0, named=True)
    worst = df.sort("ret").row(0, named=True)

    minute_cache: dict[str, pl.DataFrame] = {}

    for label, rec in [("【最高收益】", best), ("【最低收益】", worst)]:
        print(f"\n{label} {rec['symbol']} {rec['name']}  "
              f"信号日 {rec['signal_date']}  当日排名 {rec['rank']}/{rec['total']}")
        print(f"  买入日 {rec['buy_date']} 开盘价 {rec['entry_price']:.2f} -> "
              f"卖出日 {rec['sell_date']} 收盘价 {rec['exit_price']:.2f}  "
              f"收益 {rec['ret']*100:+.2f}%")
        buy_path = _minute_path(minute_cache, rec["symbol"], rec["buy_date"])
        if buy_path:
            _print_path(f"买入日 {rec['buy_date']} 分钟路径", buy_path, rec["entry_price"])
        if rec["sell_date"] in minute_days:
            sell_path = _minute_path(minute_cache, rec["symbol"], rec["sell_date"])
            if sell_path:
                _print_path(f"卖出日 {rec['sell_date']} 分钟路径", sell_path, rec["entry_price"])


def main() -> None:
    ap = argparse.ArgumentParser(description="分析策略候选个股的最高/最低收益路径")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--hold-days", type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(args.strategy, args.hold_days)


if __name__ == "__main__":
    main()
