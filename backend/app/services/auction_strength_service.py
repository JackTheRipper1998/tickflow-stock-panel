"""集合竞价强弱判断 — 开盘涨幅类型 + 集合竞价成交量强弱, 用于判断
"今天开盘到底是真强还是假强", 供盘中实时策略加一层开盘确认/风险过滤。

集合竞价量的获取(2026-07-04 用 2026-07-03 全市场分钟K实测校准过):
  数据源的分钟K一天有 241 根, 第一根时间戳 09:30, 其中 99.82% 的股票该根
  O=H=L=C(集合竞价所有成交在同一价格撮合, 必然四价合一)——
  即【09:30 这根分钟K本身就是集合竞价的成交量, 是精确值, 不是近似】。
  注意: 不能用"09:31时刻的累计量"当竞价量 —— 实测竞价量中位数只占
  (竞价+连续竞价第一分钟)的 18%, 那样会中位数高估 5.5 倍(P90 高达 16.7 倍)。

  盘中获取路径(两条腿, 先到先用, 精确值覆盖近似值):
    1. 09:26 盘前快照(snapshot_premarket): 竞价 09:25 撮合完、09:30 连续竞价
       尚未开始, 此刻实时行情的当日累计成交量 = 纯集合竞价量。
       带陈旧数据防护: 若某票快照量 > 前一日全天量的 80%, 几乎必然是行情源
       还没刷新、返回的是昨天的全天累计量(真实竞价量不可能到前一日的80%),
       丢弃该票等 09:32 校正。
       (注: 依赖行情源在盘前时段就发布竞价撮合量, 若行情源盘前不给量,
        该路径当天为空, 由第2条腿兜底。)
    2. 09:32 分钟K校正(snapshot_from_minute_k): 直接调 API 拉当天 09:30 那根
       竞价K线(全市场按批, 窗口只有几分钟, 载荷很小), 拿到的是精确值,
       覆盖第1条腿的结果。若 API 盘中不提供当天分钟K则该路径为空, 保留第1条腿。

两个核心指标:
  1. 涨幅类型(gap_type): 今日开盘价 vs 昨收 —
     大高开(>5%) / 小高开(3%~5%) / 高开(1%~3%) / 平开(-1%~1%) /
     低开(-3%~-1%) / 小低开(-5%~-3%) / 大低开(<-5%)
     不需要快照, 只要 open/prev_close 都在就随时可算。
  2. 成交量类型(volume_type): 集合竞价成交量 / 前一交易日全天成交量 —
     极致缩量(<0.2%) / 缩量(0.2%~0.4%) / 标准量(0.4%~0.7%, 参考位0.5%) /
     放量(0.7%~3%, 黄金区1.2%~3%) / 巨量(>=3%)
     需要竞价量快照, 最早 09:26 后可用。
     档位为数据标定值(2026-04~07 共43个交易日、18.2万样本, 剔ST):
     全市场中位数0.40%, 大盘股(≥200亿)中位数0.49%且4月/6月两段行情几乎不漂移;
     开盘买入→收盘的胜率在1.2%~3%区间最高(大盘股56%~60%), >3%优势消失,
     >5%高开时明显转差(即"放过头=出货嫌疑"成立, 但触发线是5%而不是经验值50%)。
     用户原始经验档位(标准量10%/放量20%/巨量40%)约为实测值的20倍, 已按数据校正。

组合解读(方向来自用户经验规则, 触发线按上述43日样本标定):
  - signal_auction_safe_high_open: 高开5%~8% 且竞价放量至黄金区(1.2%~3%) → 相对安全
  - signal_auction_distribution_risk: 高开>5% 且竞价放量>=5% → 出货嫌疑, 不建议参与
  - signal_auction_bottom_reversal: 大低开(<-5%) 且竞价放量5%~10% → 博反弹信号
    (该条按比例折算自经验值50%~67%, 大盘股样本太少【未验证】, 仅供参考)

不知道: 监控规则引擎、策略引擎、API 路由、调度器本身、QuoteService 内部轮询细节。
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

SNAPSHOT_FILE = "auction_vol_today.parquet"
SNAPSHOT_COL = "auction_vol"
RATIO_COL = "auction_vol_ratio"
# 盘前快照陈旧数据防护线: 快照量超过前一日全天量的这个比例, 视为行情源未刷新
STALE_GUARD_RATIO = 0.80


def _snapshot_path(data_dir: Path) -> Path:
    d = data_dir / "user_data" / "auction_vol"
    d.mkdir(parents=True, exist_ok=True)
    return d / SNAPSHOT_FILE


class AuctionStrengthService:
    """集合竞价强弱判断 — 单例, main.py 启动时 set_repo 注入。"""

    def __init__(self) -> None:
        self._repo = None
        self._ratio_cache: dict[str, float] = {}
        self._ratio_cache_date: date | None = None

    def set_repo(self, repo) -> None:
        self._repo = repo

    def _invalidate_cache(self) -> None:
        self._ratio_cache = {}
        self._ratio_cache_date = None

    def _save_snapshot(self, snap: pl.DataFrame) -> bool:
        try:
            snap.select(["date", "symbol", SNAPSHOT_COL, "source"]).write_parquet(
                _snapshot_path(self._repo.store.data_dir)
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("集合竞价快照落盘失败: %s", e)
            return False

    # ================================================================
    # 第1条腿: 09:26 盘前快照 (竞价撮合后、连续竞价前, 累计量=纯竞价量)
    # ================================================================
    def snapshot_premarket(self) -> None:
        if not self._repo:
            return
        from app.market_time import cn_today
        today = cn_today()
        try:
            df, d = self._repo.get_enriched_latest()
        except Exception as e:  # noqa: BLE001
            logger.warning("盘前竞价快照获取行情失败: %s", e)
            return
        if df.is_empty() or d != today:
            logger.info("盘前竞价快照跳过: 无实时数据或非当日 (enriched_date=%s)", d)
            return
        if "volume" not in df.columns or "symbol" not in df.columns:
            return

        snap = (
            df.select(["symbol", "volume"])
            .rename({"volume": SNAPSHOT_COL})
            .filter(pl.col("symbol").is_not_null() & pl.col(SNAPSHOT_COL).is_not_null() & (pl.col(SNAPSHOT_COL) > 0))
        )
        if snap.is_empty():
            logger.info("盘前竞价快照跳过: 行情源盘前无成交量数据 (等待09:32分钟K校正)")
            return

        # 陈旧数据防护: 快照量 > 前一日全天量的80% → 视为行情源未刷新(昨天的累计量), 丢弃
        prev = self._load_prev_day_volume(self._repo.store.data_dir, today)
        if not prev.is_empty():
            before = snap.height
            snap = snap.join(prev, on="symbol", how="left").filter(
                pl.col("_prev_volume").is_null()
                | (pl.col(SNAPSHOT_COL) <= pl.col("_prev_volume") * STALE_GUARD_RATIO)
            ).drop("_prev_volume")
            dropped = before - snap.height
            if dropped:
                logger.info("盘前竞价快照: %d 只疑似陈旧数据被丢弃(等待09:32校正)", dropped)
        if snap.is_empty():
            return

        snap = snap.with_columns([
            pl.lit(str(today)).alias("date"),
            pl.lit("premarket_quote").alias("source"),
        ])
        if self._save_snapshot(snap):
            self._invalidate_cache()
            logger.info("盘前竞价快照完成: %d 只, 日期 %s", snap.height, today)

    # ================================================================
    # 第2条腿: 09:32 分钟K校正 (拉当天09:30竞价K线, 精确值, 覆盖第1条腿)
    # ================================================================
    def snapshot_from_minute_k(self) -> None:
        if not self._repo:
            return
        from app.market_time import CN_TZ, cn_today
        from app.services.kline_sync import sync_minute_batch

        today = cn_today()
        try:
            inst = self._repo.get_instruments()
            symbols = inst["symbol"].to_list() if not inst.is_empty() else []
        except Exception as e:  # noqa: BLE001
            logger.warning("竞价K线校正获取股票列表失败: %s", e)
            return
        if not symbols:
            return

        start = datetime(today.year, today.month, today.day, 9, 25, tzinfo=CN_TZ)
        end = datetime(today.year, today.month, today.day, 9, 31, tzinfo=CN_TZ)
        try:
            bars = sync_minute_batch(symbols, start_time=start, end_time=end, batch_size=100)
        except Exception as e:  # noqa: BLE001
            logger.warning("竞价K线校正拉取失败(保留盘前快照结果): %s", e)
            return
        if bars.is_empty() or "datetime" not in bars.columns:
            logger.info("竞价K线校正: API 未返回当天分钟K(保留盘前快照结果)")
            return

        # 分钟K datetime 为 UTC-naive: 09:30 CN = 01:30 UTC
        t930_utc = datetime(today.year, today.month, today.day, 1, 30)
        snap = (
            bars.filter(pl.col("datetime") == t930_utc)
            .select(["symbol", pl.col("volume").alias(SNAPSHOT_COL)])
            .filter(pl.col("symbol").is_not_null() & pl.col(SNAPSHOT_COL).is_not_null() & (pl.col(SNAPSHOT_COL) > 0))
        )
        if snap.is_empty():
            logger.info("竞价K线校正: 无当天09:30竞价K线(保留盘前快照结果)")
            return

        snap = snap.with_columns([
            pl.lit(str(today)).alias("date"),
            pl.lit("minute_k").alias("source"),
        ])
        if self._save_snapshot(snap):
            self._invalidate_cache()
            logger.info("竞价K线校正完成(精确值): %d 只, 日期 %s", snap.height, today)

    # ================================================================
    # 比值计算 (供 quote_service 注入 enriched, 当天只算一次)
    # ================================================================
    def get_ratio_map(self, today: date) -> dict[str, float]:
        """返回 {symbol: auction_vol_ratio}。当天只算一次, 内存缓存复用。"""
        if self._ratio_cache_date == today and self._ratio_cache:
            return self._ratio_cache
        if not self._repo:
            return {}
        try:
            ratio_map = self._compute_ratio_map(today)
        except Exception as e:  # noqa: BLE001
            logger.warning("集合竞价量能比值计算失败: %s", e)
            return {}
        if ratio_map:
            self._ratio_cache = ratio_map
            self._ratio_cache_date = today
        return ratio_map

    def _compute_ratio_map(self, today: date) -> dict[str, float]:
        data_dir = self._repo.store.data_dir
        snap_path = _snapshot_path(data_dir)
        if not snap_path.exists():
            return {}
        snap = pl.read_parquet(snap_path)
        snap = snap.filter(pl.col("date") == str(today))
        if snap.is_empty():
            # 今日尚未快照(09:26前, 或行情服务今天才开启)
            return {}

        prev_volume = self._load_prev_day_volume(data_dir, today)
        if prev_volume.is_empty():
            return {}

        merged = snap.join(prev_volume, on="symbol", how="inner").filter(pl.col("_prev_volume") > 0)
        if merged.is_empty():
            return {}
        merged = merged.with_columns((pl.col(SNAPSHOT_COL) / pl.col("_prev_volume")).alias(RATIO_COL))
        return dict(zip(merged["symbol"].to_list(), merged[RATIO_COL].to_list()))

    @staticmethod
    def _load_prev_day_volume(data_dir: Path, today: date) -> pl.DataFrame:
        """前一交易日全天成交量 — 直接扫原始日线表(不依赖实时 enriched 缓存,
        避免缓存已被当天数据覆盖导致取不到"昨天"的问题)。
        单位与分钟K一致(2026-07-03 实测分钟K全天求和 == 日线量, 无换算)。"""
        daily_glob = str(data_dir / "kline_daily" / "**" / "*.parquet")
        try:
            hist = (
                pl.scan_parquet(daily_glob)
                .filter(pl.col("date") < today)
                .select(["symbol", "date", "volume"])
                .collect()
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("读取前一交易日成交量失败: %s", e)
            return pl.DataFrame()
        if hist.is_empty():
            return pl.DataFrame()
        last_date = hist["date"].max()
        return (
            hist.filter(pl.col("date") == last_date)
            .select(["symbol", "volume"])
            .rename({"volume": "_prev_volume"})
        )

    # ================================================================
    # 字段注入 (供 quote_service._flush_live_enriched 调用)
    # ================================================================
    @staticmethod
    def inject_gap_fields(df: pl.DataFrame) -> pl.DataFrame:
        """涨幅类型: 不需要快照, 随时可用 open/prev_close 算。"""
        if df.is_empty() or "open" not in df.columns or "prev_close" not in df.columns:
            return df
        df = df.with_columns(
            pl.when(pl.col("prev_close") > 0)
            .then(pl.col("open") / pl.col("prev_close") - 1.0)
            .otherwise(None)
            .alias("gap_pct"),
        )
        gap_bucket = (
            pl.when(pl.col("gap_pct") > 0.05).then(pl.lit("大高开"))
            .when(pl.col("gap_pct") > 0.03).then(pl.lit("小高开"))
            .when(pl.col("gap_pct") > 0.01).then(pl.lit("高开"))
            .when(pl.col("gap_pct") >= -0.01).then(pl.lit("平开"))
            .when(pl.col("gap_pct") >= -0.03).then(pl.lit("低开"))
            .when(pl.col("gap_pct") >= -0.05).then(pl.lit("小低开"))
            .otherwise(pl.lit("大低开"))
        )
        return df.with_columns(
            pl.when(pl.col("gap_pct").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(gap_bucket)
            .alias("gap_type"),
        )

    @staticmethod
    def inject_volume_fields(df: pl.DataFrame) -> pl.DataFrame:
        """成交量类型 + 3个组合信号: 调用前需先 JOIN 好 auction_vol_ratio 列。"""
        if df.is_empty() or RATIO_COL not in df.columns:
            return df
        # 档位为43日样本标定值(见文件头), 不是用户原始经验值(那套约大20倍)
        vol_bucket = (
            pl.when(pl.col(RATIO_COL) < 0.002).then(pl.lit("极致缩量"))
            .when(pl.col(RATIO_COL) < 0.004).then(pl.lit("缩量"))
            .when(pl.col(RATIO_COL) < 0.007).then(pl.lit("标准量"))
            .when(pl.col(RATIO_COL) < 0.03).then(pl.lit("放量"))
            .otherwise(pl.lit("巨量"))
        )
        df = df.with_columns(
            pl.when(pl.col(RATIO_COL).is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(vol_bucket)
            .alias("volume_type"),
        )
        if "gap_pct" not in df.columns:
            return df
        # 触发线为43日样本标定值(见文件头); bottom_reversal 按比例折算, 未验证
        return df.with_columns([
            (
                (pl.col("gap_pct") >= 0.05) & (pl.col("gap_pct") <= 0.08)
                & (pl.col(RATIO_COL) >= 0.012) & (pl.col(RATIO_COL) < 0.03)
            ).fill_null(False).alias("signal_auction_safe_high_open"),
            (
                (pl.col("gap_pct") > 0.05) & (pl.col(RATIO_COL) >= 0.05)
            ).fill_null(False).alias("signal_auction_distribution_risk"),
            (
                (pl.col("gap_pct") < -0.05)
                & (pl.col(RATIO_COL) >= 0.05) & (pl.col(RATIO_COL) <= 0.10)
            ).fill_null(False).alias("signal_auction_bottom_reversal"),
        ])


auction_strength_service = AuctionStrengthService()
