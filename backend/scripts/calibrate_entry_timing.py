#!/usr/bin/env python
"""校准策略买入时机 — 用历史分钟K比较「次日不同时点买入」对次日收益的影响。

背景：strategy_id 策略的候选层在信号日(T)收盘后产生，实盘只能在次日(T+1)
才能真正下单；但 T+1 当天究竟几点买入（开盘/日中/尾盘），会显著影响持有
max_hold_days 天后的实际收益 —— 尤其对"追涨"类策略，越晚介入往往越差。

方法：
    1. 用 StrategyEngine._load_file 加载真实策略文件（basic_filter/filter_fn/
       scoring/limit 与线上完全一致，不手工复制规则，避免和策略文件跑偏）。
    2. 对分钟K已覆盖的每个信号日 T，复现候选层 + 评分 + limit 截断，得到当日
       真实会入选的候选股。
    3. 对每个候选，在 T+1 的分钟K里取多个候选入场时点的价格，与 T+1+hold_days
       日K收盘价比较，算出不同入场时点的平均/中位收益与胜率。

用法（从 backend/ 目录运行）：
    .venv/bin/python -m scripts.calibrate_entry_timing --strategy ai_20260627
    .venv/bin/python -m scripts.calibrate_entry_timing --strategy ai_20260627 \
        --times 09:50 13:10 14:55 --hold-days 1

分钟K数据有限（本地按 minute_sync_days 滚动保留），样本天数取决于
data/kline_minute 实际覆盖的交易日区间；结果仅反映该窗口的市场结构，
需要随着分钟K积累定期重跑复核。
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
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
    raise FileNotFoundError(f"策略文件未找到: {strategy_id}.py (搜索目录: {STRATEGY_DIRS})")


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


def _score_and_truncate(hits: pl.DataFrame, scoring: dict[str, float], limit: int) -> pl.DataFrame:
    if not scoring or hits.is_empty():
        return hits.head(limit)
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
        return hits.head(limit)
    return hits.with_columns((score_expr * 100).alias("_score")).sort("_score", descending=True).head(limit)


def _bj_to_utc(d: str, hh_mm: str) -> datetime:
    h, m = map(int, hh_mm.split(":"))
    return datetime.strptime(d, "%Y-%m-%d").replace(hour=(h - 8) % 24, minute=m)


def run(strategy_id: str, times: list[str], hold_days: int) -> None:
    from app.strategy.engine import StrategyEngine

    strategy_path = _find_strategy_file(strategy_id)
    strategy = StrategyEngine._load_file(strategy_path)
    logger.info("策略: %s (%s)  scoring=%s  limit=%d", strategy.meta.get("name"), strategy_id,
                strategy.meta.get("scoring"), strategy.meta.get("limit", 50))

    enriched = _load_daily_panel()
    all_days = sorted(enriched.select("date").unique().to_series().cast(pl.Utf8).to_list())
    minute_days = _minute_trading_days()
    logger.info("分钟K覆盖交易日 (%d 天): %s", len(minute_days), minute_days)

    basic_expr = StrategyEngine._basic_filter_expr(enriched, strategy.basic_filter)
    scoring = strategy.meta.get("scoring", {})
    limit = int(strategy.meta.get("limit", 50))

    candidates = []
    for d in minute_days:
        if d not in all_days:
            continue
        di = all_days.index(d)
        if di + hold_days + 1 >= len(all_days):
            continue
        buy_date = all_days[di + 1]
        sell_date = all_days[di + 1 + hold_days]
        if buy_date not in minute_days:
            continue  # 买入日没有分钟K, 无法算精确入场价

        day_df = enriched.filter(pl.col("date") == datetime.strptime(d, "%Y-%m-%d").date())
        if day_df.is_empty():
            continue
        mask = day_df.select(
            ((basic_expr if basic_expr is not None else pl.lit(True)) & strategy.filter_fn(day_df, {})).alias("_hit")
        )["_hit"].fill_null(False)
        hits = day_df.filter(mask)
        if hits.is_empty():
            continue
        hits = _score_and_truncate(hits, scoring, limit)
        for row in hits.iter_rows(named=True):
            candidates.append({
                "signal_date": d, "buy_date": buy_date, "sell_date": sell_date,
                "symbol": row["symbol"],
            })

    if not candidates:
        logger.warning("无候选样本, 无法校准 (分钟K覆盖窗口内没有触发信号)")
        return
    cand_df = pl.DataFrame(candidates)
    logger.info("候选样本数: %d  (信号日: %s)", cand_df.height, sorted(set(cand_df["signal_date"].to_list())))

    daily_close = enriched.select(["symbol", "date", "close"]).with_columns(pl.col("date").cast(pl.Utf8))
    minute_cache: dict[str, pl.DataFrame] = {}
    results = []

    for row in cand_df.iter_rows(named=True):
        buy_date = row["buy_date"]
        if buy_date not in minute_cache:
            minute_cache[buy_date] = pl.read_parquet(
                str(DATA_DIR / "kline_minute" / f"date={buy_date}" / "part.parquet")
            ).sort("datetime")
        mdf = minute_cache[buy_date]
        sym = row["symbol"]
        sub = mdf.filter(pl.col("symbol").str.starts_with(sym))
        if sub.is_empty() or sub["open"][0] is None:
            continue

        sell_row = daily_close.filter((pl.col("symbol") == sym) & (pl.col("date") == row["sell_date"]))
        if sell_row.is_empty():
            continue

        rec = {"symbol": sym, "signal_date": row["signal_date"], "buy_date": buy_date,
               "open_price": float(sub["open"][0]), "sell_price": float(sell_row["close"][0])}
        for t in times:
            target_dt = _bj_to_utc(buy_date, t)
            s = sub.filter(pl.col("datetime") <= target_dt)
            rec[f"px_{t.replace(':', '')}"] = float(s["close"][-1]) if not s.is_empty() else None
        results.append(rec)

    res_df = pl.DataFrame(results)
    logger.info("有效样本数(买入日+卖出日数据齐全): %d", res_df.height)

    price_cols = ["open_price"] + [f"px_{t.replace(':', '')}" for t in times]
    print(f"\n{'入场时点':<14}{'样本':>6}{'均值收益':>10}{'中位收益':>10}{'胜率':>8}")
    for col in price_cols:
        if col not in res_df.columns:
            continue
        sub = res_df.filter(pl.col(col).is_not_null() & (pl.col(col) > 0))
        if sub.is_empty():
            continue
        ret = (sub["sell_price"] - sub[col]) / sub[col]
        label = "次日开盘价" if col == "open_price" else f"次日{col[3:5]}:{col[5:7]}"
        print(f"{label:<14}{sub.height:>6}{ret.mean()*100:>9.2f}%{ret.median()*100:>9.2f}%{(ret > 0).sum() / len(ret) * 100:>7.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description="用历史分钟K校准策略的买入时机")
    ap.add_argument("--strategy", required=True, help="策略 id (对应 strategies/*/<id>.py 文件名)")
    ap.add_argument("--times", nargs="+", default=["09:50", "13:10", "14:55"],
                     help="候选入场时点 (北京时间 HH:MM), 默认: 09:50 13:10 14:55 (日中/尾盘)")
    ap.add_argument("--hold-days", type=int, default=1, help="持有交易日数, 默认 1 (对应 MAX_HOLD_DAYS=1 的追涨策略)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(args.strategy, args.times, args.hold_days)


if __name__ == "__main__":
    main()
