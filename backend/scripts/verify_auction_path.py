#!/usr/bin/env python
"""竞价路径采集数据 QA — 独立对照重算 (预注册清单·立即执行项3)。

方法论参考当年发现「09:31累计量高估竞价量5.5倍」的做法: 不信任字段语义,
用已知可靠的独立数据源交叉验证, 实证后才允许因子使用该字段。

检查项:
  Q1. 09:25:15 终值拍的匹配价(买一价) vs 当日日线开盘价 — 应完全一致
      (竞价撮合价即开盘价)。不一致率>1%则该字段不可用于 F3。
  Q2. 09:25:15 quote_cum_vol vs 当日09:30竞价K线量(已实证的精确竞价量) —
      比值应≈1。系统性偏离则 F6/F7 的量路径不可信。
  Q3. 竞价期间(09:20~09:25)买一价==卖一价的比例 — A股竞价盘口应显示虚拟
      匹配价(两侧同价)。比例低说明行情源盘前给的不是竞价撮合视图, F3/F6 定义
      需按实际语义修订(修订属"追加", 需在预注册文档补记, 不改原定义)。
  Q4. 09:16~09:19(可撤单段) vs 09:20~09:24(不可撤段)的盘口结构差异 + 每拍
      覆盖率/空值率 — F7 的挂单基线是否可用。
  Q5. 逐拍时间戳与行数完整性 (应有10拍)。

用法 (backend/): .venv/Scripts/python.exe -m scripts.verify_auction_path [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="验证日期, 默认最新采集日")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    base = DATA_DIR / "user_data" / "auction_path"
    days = sorted(p.name.split("=")[1] for p in base.glob("date=*")) if base.exists() else []
    if not days:
        print("尚无竞价路径数据 — 采集器于首个交易日 09:16 开始工作, 之后再跑本脚本。")
        return
    day = args.date or days[-1]
    fp = base / f"date={day}" / "part.parquet"
    if not fp.exists():
        print(f"{day} 无数据。已有: {days}")
        return
    df = pl.read_parquet(fp)
    print(f"\n=== 竞价路径 QA: {day} ===  行数={df.height}, 标的={df['symbol'].n_unique()}, "
          f"拍数={df['ts'].n_unique()}")

    # Q5: 拍完整性
    ticks = sorted(df["ts"].unique().to_list())
    print(f"\nQ5 拍列表 ({len(ticks)}/10): {ticks}")
    per_tick = df.group_by("ts").agg(
        pl.len().alias("n"),
        pl.col("bid_p1").null_count().alias("bid_p1_null"),
        pl.col("quote_cum_vol").null_count().alias("vol_null"),
    ).sort("ts")
    print(per_tick)

    final_ts = [t for t in ticks if t >= "09:25"]
    if not final_ts:
        print("\n无 09:25 终值拍, Q1/Q2 跳过 (检查采集器调度)")
        return
    final = df.filter(pl.col("ts") == final_ts[0])

    # Q1: 终值匹配价 vs 日线开盘价
    daily = (
        pl.scan_parquet(str(DATA_DIR / "kline_daily" / f"date={day}" / "*.parquet"))
        .select("symbol", pl.col("open").alias("daily_open"))
        .collect()
    ) if (DATA_DIR / "kline_daily" / f"date={day}").exists() else pl.DataFrame(
        schema={"symbol": pl.Utf8, "daily_open": pl.Float64})
    if daily.is_empty():
        print("\nQ1 跳过: 当日日线未落盘 (盘后再跑)")
    else:
        q1 = final.join(daily, on="symbol", how="inner").with_columns(
            ((pl.col("bid_p1") - pl.col("daily_open")).abs() > 0.005).alias("mismatch"),
        )
        mm = q1.filter(pl.col("mismatch"))
        rate = mm.height / q1.height * 100 if q1.height else float("nan")
        print(f"\nQ1 终值匹配价 vs 日线开盘价: n={q1.height}, 不一致率={rate:.1f}% "
              f"({'PASS' if rate <= 1.0 else 'FAIL — F3 字段语义需修订'})")
        if mm.height:
            print(mm.select("symbol", "bid_p1", "daily_open").head(5))

    # Q2: 终值累计量 vs 09:30 竞价K线量
    mk = DATA_DIR / "kline_minute" / f"date={day}" / "part.parquet"
    if not mk.exists():
        print("\nQ2 跳过: 当日分钟K未落盘 (盘后再跑)")
    else:
        auc = (
            pl.scan_parquet(str(mk))
            .filter(pl.col("datetime").dt.time() == dt.time(1, 30))
            .select("symbol", pl.col("volume").alias("auction_k_vol"))
            .collect()
        )
        q2 = final.join(auc, on="symbol", how="inner").filter(
            pl.col("quote_cum_vol").is_not_null() & (pl.col("auction_k_vol") > 0)
        ).with_columns((pl.col("quote_cum_vol") / pl.col("auction_k_vol")).alias("ratio"))
        if q2.is_empty():
            print("\nQ2: 无可比样本 (quote_cum_vol 全空? 行情源盘前不给量, F6/F7 量路径降级)")
        else:
            print(f"\nQ2 终值累计量/竞价K线量: n={q2.height}, 中位={q2['ratio'].median():.3f}, "
                  f"P10={q2['ratio'].quantile(0.1):.3f}, P90={q2['ratio'].quantile(0.9):.3f} "
                  f"(中位∈[0.95,1.05] 为 PASS)")

    # Q3: 竞价段 买一==卖一 比例
    match_seg = df.filter((pl.col("ts") >= "09:20") & pl.col("bid_p1").is_not_null() & pl.col("ask_p1").is_not_null())
    if match_seg.height:
        same = match_seg.filter((pl.col("bid_p1") - pl.col("ask_p1")).abs() < 0.005)
        print(f"\nQ3 竞价段(≥09:20)买一==卖一比例: {same.height}/{match_seg.height} "
              f"= {same.height/match_seg.height*100:.1f}% "
              f"(高=竞价撮合视图, F3可用; 低=普通盘口视图, F3/F6语义需修订)")
    else:
        print("\nQ3: 竞价段无有效盘口数据")

    # Q4: 可撤单段 vs 不可撤段 盘口结构
    early = df.filter(pl.col("ts") < "09:20")
    late = df.filter((pl.col("ts") >= "09:20") & (pl.col("ts") < "09:25"))
    for label, seg in [("09:16~09:19(可撤)", early), ("09:20~09:24(不可撤)", late)]:
        if seg.height:
            valid = seg.filter(pl.col("bid_p1").is_not_null())
            spread = valid.filter(pl.col("ask_p1").is_not_null()).with_columns(
                (pl.col("ask_p1") - pl.col("bid_p1")).alias("spread"))
            print(f"Q4 {label}: 行={seg.height}, 盘口非空率={valid.height/seg.height*100:.0f}%, "
                  f"价差中位={spread['spread'].median() if spread.height else float('nan'):.3f}")
        else:
            print(f"Q4 {label}: 无数据")

    print("\n完成。首日结果无论 PASS/FAIL 都记入预注册文档附录 (字段语义实证记录)。")


if __name__ == "__main__":
    main()
