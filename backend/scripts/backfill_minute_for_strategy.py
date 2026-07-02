#!/usr/bin/env python
"""按策略候选池回填历史分钟K, 用于扩展入场时机测算的窗口。

背景：分钟K本地按 minute_sync_days 滚动保留 (上限30天), 覆盖不到3个月级
别的回测窗口。TickFlow 分钟K批量接口 (kline.minute.batch, 通常 rpm=30,
单批最多100只) 支持按 start_time/end_time 拉取任意历史交易日, 因此按需
逐日回填策略候选池当天(T+1买入日)需要的标的即可, 不必回填全市场。

用法（从 backend/ 目录运行）：
    .venv/bin/python -m scripts.backfill_minute_for_strategy --strategy ai_20260627 --days 65
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import polars as pl

from app.config import settings
from app.services import kline_sync
from app.tickflow.capabilities import Cap
from app.tickflow.policy import detect_capabilities
from app.tickflow.repository import DataStore, KlineRepository

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


def _candidate_symbols(day_df: pl.DataFrame, basic_expr, strategy, scoring: dict, limit: int) -> list[str]:
    mask = day_df.select(
        ((basic_expr if basic_expr is not None else pl.lit(True)) & strategy.filter_fn(day_df, {})).alias("_hit")
    )["_hit"].fill_null(False)
    hits = day_df.filter(mask)
    if hits.is_empty():
        return []
    score_expr = None
    for col, w in scoring.items():
        if col not in hits.columns:
            continue
        vmin, vmax = hits[col].min(), hits[col].max()
        vrange = (vmax - vmin) if (vmax is not None and vmin is not None) else 0
        normalized = ((pl.col(col) - vmin) / vrange) if vrange else pl.lit(0.5)
        part = normalized * (w / sum(scoring.values()))
        score_expr = part if score_expr is None else score_expr + part
    ranked = (
        hits.with_columns((score_expr * 100).alias("_score")).sort("_score", descending=True)
        if score_expr is not None else hits
    )
    return ranked.head(limit)["symbol"].to_list()


def run(strategy_id: str, lookback_days: int) -> None:
    from app.indicators.pipeline import compute_all
    from app.strategy.engine import StrategyEngine

    strategy_path = _find_strategy_file(strategy_id)
    strategy = StrategyEngine._load_file(strategy_path)
    scoring = strategy.meta.get("scoring", {})
    limit = int(strategy.meta.get("limit", 50))

    enriched = pl.scan_parquet(str(DATA_DIR / "kline_daily_enriched" / "**" / "*.parquet")).collect()
    inst = pl.read_parquet(str(DATA_DIR / "instruments" / "instruments.parquet"))
    enriched = compute_all(enriched, instruments=inst)

    all_days = sorted(enriched.select("date").unique().to_series().cast(pl.Utf8).to_list())
    window_days = all_days[-(lookback_days + 1):-1]  # 最后一天留作最新信号日的 T+1 buy_date
    logger.info("回填窗口: %s ~ %s (%d 个信号日)", window_days[0], window_days[-1], len(window_days))

    basic_expr = StrategyEngine._basic_filter_expr(enriched, strategy.basic_filter)

    store = DataStore(settings.data_dir)
    repo = KlineRepository(store)
    capset = detect_capabilities()
    lim = capset.limits(Cap.KLINE_MINUTE_BATCH)
    batch_size = lim.batch if lim and lim.batch else 100
    rpm = lim.rpm if lim else 30
    sleep_s = 60.0 / rpm if rpm else 2.0
    logger.info("kline.minute.batch: batch_size=%d rpm=%d -> 每次调用后 sleep %.1fs", batch_size, rpm, sleep_s)

    existing_minute_days = {p.name.split("=")[-1] for p in (DATA_DIR / "kline_minute").glob("date=*")}

    fetched, skipped, empty = 0, 0, 0
    for i, d in enumerate(window_days):
        di = all_days.index(d)
        if di + 1 >= len(all_days):
            continue
        buy_date = all_days[di + 1]
        if buy_date in existing_minute_days:
            skipped += 1
            continue

        day_df = enriched.filter(pl.col("date") == datetime.strptime(d, "%Y-%m-%d").date())
        if day_df.is_empty():
            continue
        symbols = _candidate_symbols(day_df, basic_expr, strategy, scoring, limit)
        if not symbols:
            continue

        buy_dt = datetime.strptime(buy_date, "%Y-%m-%d")
        start_time = buy_dt.replace(hour=9, minute=25)
        end_time = buy_dt.replace(hour=15, minute=5)

        logger.info("[%d/%d] 回填 %s (信号日 %s 候选 %d 只)", i + 1, len(window_days), buy_date, d, len(symbols))
        df = kline_sync.sync_minute_batch(symbols, start_time=start_time, end_time=end_time, batch_size=batch_size)
        if df.is_empty():
            logger.warning("  %s 无数据返回", buy_date)
            empty += 1
            time.sleep(sleep_s)
            continue

        df = df.with_columns(pl.col("datetime").dt.date().alias("_trade_date"))
        for day_part in df.partition_by("_trade_date"):
            trade_date = day_part["_trade_date"][0]
            out = DATA_DIR / "kline_minute" / f"date={trade_date}" / "part.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            day_part.drop("_trade_date").sort("symbol", "datetime").write_parquet(out)
            fetched += 1
            existing_minute_days.add(str(trade_date))

        time.sleep(sleep_s)

    logger.info("回填完成: 新增 %d 天, 跳过(已存在) %d 天, 空数据 %d 天", fetched, skipped, empty)


def main() -> None:
    ap = argparse.ArgumentParser(description="按策略候选池回填历史分钟K (用于扩展入场时机测算窗口)")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--days", type=int, default=65, help="回填的信号日交易日窗口长度, 默认65(约3个月)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(args.strategy, args.days)


if __name__ == "__main__":
    main()
