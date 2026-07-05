"""开盘抢筹强度 — 开盘5分钟成交量 vs 自身过去5日同时段均量。

背景: 分钟K只在盘后15:30批量同步(见 kline_sync.py), 盘中当天的分钟K不存在,
无法直接从分钟K读"今天开盘5分钟"的量。改用实时行情本身在 09:36 时刻的累计成交量
作为"开盘5分钟成交量"的近似值(09:30开盘, 09:36时累计量基本就是开盘头几分钟的量)。

职责:
  - 09:36 定时快照: 从 QuoteService 维护的实时 enriched 缓存里取当日累计成交量,
    按 symbol 落盘到小文件, 形成滚动历史(最近30个交易日)。
  - 结合历史快照(不足5天时从已同步的分钟K一次性回补), 算出
    opening_vol_ratio_5d = 今日开盘5分钟量 / 自身过去5日同时段均量。
  - 供 quote_service 把这一列 JOIN 进当日 enriched 缓存, 使其像 turnover_rate 一样
    可以直接被监控中心的 signal 规则、自定义信号、策略 scoring/filter 引用。

不知道: 监控规则引擎、策略引擎、API、调度器本身(由 daily_pipeline 的 09:36 job 调用)。
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 5
MAX_HISTORY_DAYS = 30
HISTORY_FILE = "opening_vol_history.parquet"
SNAPSHOT_COL = "open5m_vol"
RATIO_COL = "opening_vol_ratio_5d"


def _history_path(data_dir: Path) -> Path:
    d = data_dir / "user_data" / "opening_vol"
    d.mkdir(parents=True, exist_ok=True)
    return d / HISTORY_FILE


def _empty_history() -> pl.DataFrame:
    return pl.DataFrame({"date": [], "symbol": [], SNAPSHOT_COL: []},
                         schema={"date": pl.Utf8, "symbol": pl.Utf8, SNAPSHOT_COL: pl.Float64})


def _load_history(data_dir: Path) -> pl.DataFrame:
    p = _history_path(data_dir)
    if not p.exists():
        return _empty_history()
    try:
        return pl.read_parquet(p)
    except Exception as e:  # noqa: BLE001
        logger.warning("开盘量历史读取失败: %s", e)
        return _empty_history()


def _save_history(data_dir: Path, df: pl.DataFrame) -> None:
    try:
        df.write_parquet(_history_path(data_dir))
    except Exception as e:  # noqa: BLE001
        logger.warning("开盘量历史落盘失败: %s", e)


class OpeningSurgeService:
    """开盘抢筹强度服务 — 单例, main.py 启动时 set_repo 注入。"""

    def __init__(self) -> None:
        self._repo = None
        self._ratio_cache: dict[str, float] = {}
        self._ratio_cache_date: date | None = None

    def set_repo(self, repo) -> None:
        self._repo = repo

    # ================================================================
    # 09:36 定时快照 (由 daily_pipeline 的 scheduler job 调用)
    # ================================================================
    def snapshot_today(self) -> None:
        """快照当日开盘5分钟成交量(近似取09:36时刻的累计成交量), 落盘历史。"""
        if not self._repo:
            return
        from app.market_time import cn_today
        today = cn_today()
        try:
            df, d = self._repo.get_enriched_latest()
        except Exception as e:  # noqa: BLE001
            logger.warning("开盘快照获取行情失败: %s", e)
            return
        if df.is_empty() or d != today:
            logger.info("开盘快照跳过: 无实时数据或非当日 (enriched_date=%s)", d)
            return
        if "volume" not in df.columns or "symbol" not in df.columns:
            return

        snap = (
            df.select(["symbol", "volume"])
            .rename({"volume": SNAPSHOT_COL})
            .filter(pl.col("symbol").is_not_null() & pl.col(SNAPSHOT_COL).is_not_null() & (pl.col(SNAPSHOT_COL) > 0))
            .with_columns(pl.lit(str(today)).alias("date"))
        )
        if snap.is_empty():
            logger.info("开盘快照跳过: 无有效成交量数据")
            return

        data_dir = self._repo.store.data_dir
        hist = _load_history(data_dir)
        # 幂等: 同一天重复触发(如任务补跑)只保留最后一次快照
        hist = hist.filter(pl.col("date") != str(today))
        hist = pl.concat([hist, snap.select(["date", "symbol", SNAPSHOT_COL])], how="diagonal_relaxed")
        keep_dates = sorted(hist["date"].unique().to_list())[-MAX_HISTORY_DAYS:]
        hist = hist.filter(pl.col("date").is_in(keep_dates))
        _save_history(data_dir, hist)
        # 当日快照落盘后, 之前缓存的比值(若有)已过期, 强制下次 get_ratio_map 重算
        self._ratio_cache = {}
        self._ratio_cache_date = None
        logger.info("开盘量快照完成: %d 只, 日期 %s", snap.height, today)

    # ================================================================
    # 比值计算 (供 quote_service 注入 enriched, 当天只算一次)
    # ================================================================
    def get_ratio_map(self, today: date) -> dict[str, float]:
        """返回 {symbol: opening_vol_ratio_5d}。当天只算一次, 内存缓存复用。"""
        if self._ratio_cache_date == today and self._ratio_cache:
            return self._ratio_cache
        if not self._repo:
            return {}
        try:
            ratio_map = self._compute_ratio_map(today)
        except Exception as e:  # noqa: BLE001
            logger.warning("开盘量能比值计算失败: %s", e)
            return {}
        if ratio_map:
            self._ratio_cache = ratio_map
            self._ratio_cache_date = today
        return ratio_map

    def _compute_ratio_map(self, today: date) -> dict[str, float]:
        data_dir = self._repo.store.data_dir
        hist = _load_history(data_dir)
        today_str = str(today)
        today_snap = hist.filter(pl.col("date") == today_str)
        if today_snap.is_empty():
            # 今日尚未快照(09:36前, 或行情服务未开启)
            return {}

        past = hist.filter(pl.col("date") != today_str)
        past_dates = sorted(past["date"].unique().to_list())
        if len(past_dates) < LOOKBACK_DAYS:
            # 历史快照不足5天(功能刚上线): 从已同步的分钟K一次性回补
            past = self._backfill_from_minute_k(data_dir, today_str, LOOKBACK_DAYS)
            past_dates = sorted(past["date"].unique().to_list()) if not past.is_empty() else []
            if past.is_empty():
                return {}

        use_dates = past_dates[-LOOKBACK_DAYS:]
        past = past.filter(pl.col("date").is_in(use_dates))
        if past.is_empty():
            return {}

        baseline = past.group_by("symbol").agg(pl.col(SNAPSHOT_COL).mean().alias("_baseline"))
        merged = today_snap.join(baseline, on="symbol", how="inner").filter(pl.col("_baseline") > 0)
        if merged.is_empty():
            return {}
        merged = merged.with_columns((pl.col(SNAPSHOT_COL) / pl.col("_baseline")).alias(RATIO_COL))
        return dict(zip(merged["symbol"].to_list(), merged[RATIO_COL].to_list()))

    @staticmethod
    def _backfill_from_minute_k(data_dir: Path, today_str: str, lookback_days: int) -> pl.DataFrame:
        """历史快照不足时, 从已同步的分钟K回补过去N个交易日的开盘5分钟量(一次性, 结果落盘复用)。"""
        minute_dir = data_dir / "kline_minute"
        if not minute_dir.exists():
            return _empty_history()
        days = sorted(p.name.split("=")[-1] for p in minute_dir.glob("date=*"))
        days = [d for d in days if d < today_str][-lookback_days:]
        if not days:
            return _empty_history()

        frames = []
        for d in days:
            fp = minute_dir / f"date={d}" / "part.parquet"
            if not fp.exists():
                continue
            # 09:30-09:35 北京时间 = 01:30-01:35 UTC (分钟K以UTC存储, 见 kline_sync.py)
            t0 = datetime.strptime(d, "%Y-%m-%d").replace(hour=1, minute=30)
            t1 = datetime.strptime(d, "%Y-%m-%d").replace(hour=1, minute=35)
            try:
                day_df = (
                    pl.scan_parquet(str(fp))
                    .filter((pl.col("datetime") >= t0) & (pl.col("datetime") <= t1))
                    .group_by("symbol")
                    .agg(pl.col("volume").sum().alias(SNAPSHOT_COL))
                    .with_columns(pl.lit(d).alias("date"))
                    .collect()
                )
                if not day_df.is_empty():
                    frames.append(day_df)
            except Exception as e:  # noqa: BLE001
                logger.debug("回补分钟K失败 %s: %s", d, e)
        if not frames:
            return _empty_history()
        backfilled = pl.concat(frames, how="diagonal_relaxed").select(["date", "symbol", SNAPSHOT_COL])
        # 回补结果落盘, 避免每次都重新扫分钟K
        try:
            existing = _load_history(data_dir)
            merged = pl.concat([existing, backfilled], how="diagonal_relaxed").unique(
                subset=["date", "symbol"], keep="last"
            )
            _save_history(data_dir, merged)
        except Exception as e:  # noqa: BLE001
            logger.debug("回补结果落盘失败(不影响本次计算): %s", e)
        return backfilled


opening_surge_service = OpeningSurgeService()
