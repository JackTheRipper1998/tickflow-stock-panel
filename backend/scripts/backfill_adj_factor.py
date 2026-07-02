#!/usr/bin/env python
"""历史除权因子回填 — 修复 kline_daily_enriched 里未复权的除权跳空缺口。

背景：daily_pipeline 的除权因子同步过去只拉最近窗口 (15/30天), 本地
kline_daily 却覆盖约 1 年历史, 导致窗口之前发生的除权事件(分红/送转股)
从未被前复权, 在 enriched 面板里表现为一次性巨大的隔夜跳空, 会被回测/
分析脚本误判成真实的暴涨暴跌 (参见 002768.SZ 2026-06-18 的 -32% 假跳空)。

本脚本用 kline_daily 实际覆盖的完整历史区间, 对全市场标的重新拉一遍
除权因子 (sync_adj_factor 内部按 symbol+trade_date 去重合并, 不会重复),
然后只对新增/变更了除权记录的标的重算 kline_daily_enriched。

用法（从 backend/ 目录运行）：
    .venv/bin/python -m scripts.backfill_adj_factor
"""
from __future__ import annotations

import logging
from datetime import datetime

import polars as pl

from app.indicators.pipeline import run_pipeline
from app.services import kline_sync
from app.tickflow.policy import detect_capabilities
from app.tickflow.repository import DataStore, KlineRepository
from app.config import settings

logger = logging.getLogger(__name__)


def run() -> None:
    store = DataStore(settings.data_dir)
    repo = KlineRepository(store)
    capset = detect_capabilities()

    daily_glob = str(store.data_dir / "kline_daily" / "**" / "*.parquet")
    earliest = pl.scan_parquet(daily_glob).select(pl.col("date").min()).collect().item()
    symbols = sorted(
        pl.scan_parquet(daily_glob).select("symbol").unique().collect()["symbol"].to_list()
    )
    logger.info("标的数: %d, 历史区间: %s ~ 今天", len(symbols), earliest)

    start_time = datetime.combine(earliest, datetime.min.time())
    end_time = datetime.now()

    def _progress(cur: int, tot: int) -> None:
        if cur == 1 or cur % 10 == 0 or cur == tot:
            logger.info("除权因子拉取: %d/%d 批", cur, tot)

    added, affected = kline_sync.sync_adj_factor(
        symbols, repo, capset, start_time=start_time, end_time=end_time, on_chunk_done=_progress,
    )
    logger.info("除权因子回填完成: 新增 %d 行, 受影响标的 %d 只", added, len(affected))

    if affected:
        written = run_pipeline(data_dir=store.data_dir, symbols=affected, new_dates_only=False)
        logger.info("enriched 重算完成: %d 行 (%d 只受影响标的)", written, len(affected))
        logger.info("受影响标的: %s", affected)
    else:
        logger.info("无受影响标的, 无需重算 enriched")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run()
