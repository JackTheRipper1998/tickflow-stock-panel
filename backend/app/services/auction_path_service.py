"""集合竞价路径采集 — 为预注册因子 F3/F6/F7 积累前瞻数据 (2026-07-19 起)。

背景 (docs/prereg/2026-07-19-auction-factor-preregistration.md):
  strong_momentum_auction 封版进入前瞻纸面跟踪, 预注册了 11 个待测因子。
  其中 F3(竞价价格路径形态)/F6(未匹配量不平衡度)/F7(竞价撤单行为) 依赖
  09:15~09:25 的逐分钟盘口快照 — 此前未采集, 本服务从今天起补上。
  前瞻期每缺一天, 未来测试就少一天样本。

采集设计:
  - 时点: 交易日 09:16:15 ~ 09:25:15, 每分钟一拍 (共10拍, cron second=15 避开
    分钟边界)。09:16~09:19 = 可自由挂撤单阶段(F7挂单基线); 09:20~09:24 = 只挂
    不撤阶段(F3匹配价路径); 09:25:15 = 竞价撮合定局后的终值拍。
  - 对象: 趋势层候选池 (与 strong_momentum_auction 第一层同口径, 不含市况开关 —
    被开关停手的日子数据同样有测试价值) + 上限 200 只保险丝。
  - 内容: tf.depth.batch 五档原始价格/量 (竞价期间的语义: 买一=卖一=虚拟匹配价,
    量为匹配/未匹配量 — 具体语义以首个交易日 QA 脚本实证为准, 不预设解读)
    + 实时行情帧的累计量/额 (匹配量路径, 与 09:26 竞价量快照同源)。
  - 存储: data/user_data/auction_path/date=YYYY-MM-DD/part.parquet, 扁平列,
    只存原始值不做因子加工 (预注册纪律: 解读留给触发条件满足后的测试)。
  - QA: scripts/verify_auction_path.py 独立对照重算 (09:25终值 vs 日线开盘价、
    累计量 vs 09:30竞价K线等), 参考当年发现 09:31 口径高估5.5倍的方法。
"""
from __future__ import annotations

import logging
import threading
from datetime import date

import polars as pl

from app.market_time import cn_now, cn_today
from app.tickflow.capabilities import Cap

logger = logging.getLogger(__name__)

MAX_SYMBOLS = 200          # 保险丝: 候选池异常膨胀时截断(按成交额降序保留)
TREND_MCAP_MIN = 20e9
TREND_FLOAT_MIN = 10e9


class AuctionPathService:
    """竞价路径采集 — 单例, main.py 启动时注入。"""

    def __init__(self) -> None:
        self._repo = None
        self._app_state = None
        self._lock = threading.Lock()
        self._day_rows: list[dict] = []
        self._day: date | None = None
        self._symbols_cache: tuple[date, list[str]] | None = None

    def set_repo(self, repo) -> None:
        self._repo = repo

    def set_app_state(self, app_state) -> None:
        self._app_state = app_state

    # ── 候选池 (每天只算一次) ─────────────────────────────
    def _trend_symbols(self, today: date) -> list[str]:
        if self._symbols_cache and self._symbols_cache[0] == today:
            return self._symbols_cache[1]

        from pathlib import Path

        from app.indicators.pipeline import compute_indicators

        data_dir = Path(self._repo.store.data_dir)
        base = data_dir / "kline_daily_enriched"
        parts = sorted(base.glob("date=*"))[-90:]   # 90个分区足够算60日窗口指标
        if not parts:
            return []
        panel = pl.concat(
            [pl.scan_parquet(str(p / "*.parquet")) for p in parts],
            how="diagonal_relaxed",
        ).collect()
        panel = compute_indicators(
            panel,
            needed={"ma5", "ma10", "ma20", "momentum_5d", "momentum_10d",
                    "high_60d", "vol_ratio_5d"},
        )
        last_date = panel["date"].max()
        snap = panel.filter(pl.col("date") == last_date)

        inst = pl.read_parquet(str(data_dir / "instruments" / "instruments.parquet"))
        join_cols = [c for c in ["symbol", "name", "total_shares", "float_shares"] if c in inst.columns]
        snap = snap.join(inst.select(join_cols).unique(subset=["symbol"]), on="symbol", how="left")

        cond = (
            (pl.col("close") >= 3) & (pl.col("close") <= 300)
            & (pl.col("amount") >= 0.8e8)
            & (pl.col("close") * pl.col("total_shares") >= TREND_MCAP_MIN)
            & (pl.col("close") * pl.col("float_shares") >= TREND_FLOAT_MIN)
            & (pl.col("ma5") > pl.col("ma10")) & (pl.col("ma10") > pl.col("ma20"))
            & (pl.col("close") > pl.col("ma20"))
            & (pl.col("momentum_5d") >= 0.05)
            & (pl.col("momentum_10d") >= 0.08)
            & (pl.col("close") >= pl.col("high_60d") * 0.90)
            & (pl.col("vol_ratio_5d") >= 0.70)
            & ~pl.col("symbol").str.starts_with("688")
            & ~pl.col("symbol").str.starts_with("300")
            & ~pl.col("symbol").str.starts_with("301")
            & ~pl.col("symbol").str.ends_with(".BJ")
        )
        if "name" in snap.columns:
            cond = cond & ~pl.col("name").str.contains("(?i)ST|\\*ST|退")
        hits = (
            snap.filter(cond.fill_null(False))
            .sort("amount", descending=True)
            .head(MAX_SYMBOLS)
        )
        syms = hits["symbol"].to_list()
        self._symbols_cache = (today, syms)
        logger.info("竞价路径采集: 趋势层候选 %d 只 (基准日=%s)", len(syms), last_date)
        return syms

    # ── 单拍采集 (scheduler 09:16~09:25 每分钟调用) ──────
    def collect_tick(self) -> None:
        if not self._repo:
            return
        capset = getattr(self._app_state, "capabilities", None) if self._app_state else None
        if capset is not None and not capset.has(Cap.DEPTH5_BATCH):
            return

        today = cn_today()
        now = cn_now()
        ts = now.strftime("%H:%M:%S")
        try:
            symbols = self._trend_symbols(today)
        except Exception as e:  # noqa: BLE001
            logger.warning("竞价路径采集: 候选池计算失败: %s", e)
            return
        if not symbols:
            return

        # 五档快照
        try:
            from app.tickflow.client import get_client
            depth_data = get_client().depth.batch(symbols)
        except Exception as e:  # noqa: BLE001
            logger.warning("竞价路径采集: depth.batch 失败 (%s): %s", ts, e)
            return
        if not depth_data:
            logger.info("竞价路径采集: depth 返回空 (%s) — 行情源盘前可能不给盘口", ts)
            return

        # 实时行情帧累计量 (匹配量路径, 与 09:26 快照同源; 拿不到就置空)
        vol_map: dict[str, tuple] = {}
        try:
            df, d = self._repo.get_enriched_latest()
            if d == today and not df.is_empty() and "volume" in df.columns:
                cols = ["symbol", "volume"]
                if "amount" in df.columns:
                    cols.append("amount")
                for r in df.select(cols).iter_rows():
                    vol_map[r[0]] = (r[1], r[2] if len(r) > 2 else None)
        except Exception:  # noqa: BLE001
            pass

        rows = []
        for sym, dd in depth_data.items():
            bp = (dd.get("bid_prices") or []) + [None] * 5
            ap = (dd.get("ask_prices") or []) + [None] * 5
            bv = (dd.get("bid_volumes") or []) + [None] * 5
            av = (dd.get("ask_volumes") or []) + [None] * 5
            qv = vol_map.get(sym, (None, None))
            row = {
                "date": today.isoformat(), "ts": ts, "symbol": sym,
                "src_timestamp": dd.get("timestamp"),
                "quote_cum_vol": qv[0], "quote_cum_amount": qv[1],
            }
            for i in range(5):
                row[f"bid_p{i+1}"] = bp[i]
                row[f"bid_v{i+1}"] = bv[i]
                row[f"ask_p{i+1}"] = ap[i]
                row[f"ask_v{i+1}"] = av[i]
            rows.append(row)

        with self._lock:
            if self._day != today:
                self._day = today
                self._day_rows = []
            self._day_rows.extend(rows)
            self._persist_locked(today)
        logger.info("竞价路径采集: %s 拍到 %d 只 (当日累计 %d 行)", ts, len(rows), len(self._day_rows))

    def _persist_locked(self, today: date) -> None:
        """每拍全量重写当日 parquet (行数级别小, 崩溃最多丢当拍)。"""
        try:
            from pathlib import Path
            out_dir = Path(self._repo.store.data_dir) / "user_data" / "auction_path" / f"date={today.isoformat()}"
            out_dir.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(self._day_rows).write_parquet(out_dir / "part.parquet")
        except Exception as e:  # noqa: BLE001
            logger.warning("竞价路径采集: 落盘失败: %s", e)


auction_path_service = AuctionPathService()
