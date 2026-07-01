"""
分析「为了生活为了家」每日荐股的选股逻辑。
对每条推荐记录，取推荐日前 40 个交易日的日K数据，
计算趋势/量能指标，最后汇总共同规律。
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

from datetime import datetime
import pandas as pd
import numpy as np

# 使用项目 venv 里的 tickflow
from tickflow import TickFlow

# ── 推荐记录（日期, 名称, 代码） ──────────────────────────────────────────
PICKS = [
    ("2026-06-26", "火炬电子",   "603678.SH"),
    ("2026-06-25", "中天科技",   "600522.SH"),
    ("2026-06-24", "冰轮环境",   "000811.SZ"),
    ("2026-06-23", "烽火通信",   "600498.SH"),
    ("2026-06-22", "金钼股份",   "601958.SH"),
    ("2026-06-18", "深桑达A",    "000032.SZ"),
    ("2026-06-12", "天创时尚",   "603608.SH"),
    ("2026-06-11", "百合花",     "603823.SH"),
    ("2026-06-09", "泰和新材",   "002254.SZ"),
    ("2026-06-09", "亨通光电",   "600487.SH"),
    ("2026-06-09", "长飞光纤",   "601869.SH"),
    ("2026-06-08", "沃格光电",   "603773.SH"),
    ("2026-05-26", "露笑科技",   "002617.SZ"),
    ("2026-05-21", "利通电子",   "603629.SH"),
    ("2026-05-20", "北自科技",   "603082.SH"),
    ("2026-05-19", "京能电力",   "600578.SH"),
    ("2026-05-18", "华电辽能",   "600396.SH"),
    ("2026-05-14", "盛视科技",   "002990.SZ"),
    ("2026-05-13", "宝鼎科技",   "002552.SZ"),
    ("2026-05-11", "杭电股份",   "603618.SH"),
    ("2026-05-07", "永杉锂业",   "603399.SH"),
    ("2026-04-30", "博云新材",   "002297.SZ"),
    ("2026-04-27", "远东股份",   "600869.SH"),
    ("2026-04-24", "永鼎股份",   "600105.SH"),
    ("2026-04-24", "神剑股份",   "002361.SZ"),
    ("2026-04-21", "神剑股份",   "002361.SZ"),
    ("2026-04-17", "奥瑞德",     "600666.SH"),
    ("2026-04-16", "利通电子",   "603629.SH"),
    ("2026-04-14", "圣阳股份",   "002580.SZ"),
    ("2026-04-09", "华盛昌",     "002980.SZ"),
    ("2026-04-08", "华盛昌",     "002980.SZ"),
    ("2026-04-03", "益佰制药",   "600594.SH"),
    ("2026-04-02", "凯莱英",     "002821.SZ"),
]

def get_kline_before(tf, symbol, pick_date_str, n=40):
    """取推荐日期之前 n 根日K（不含推荐当天）。"""
    pick_ts = int(datetime.strptime(pick_date_str, "%Y-%m-%d").timestamp() * 1000)
    df = tf.klines.get(symbol, count=n + 10, end_time=pick_ts - 1, as_dataframe=True)
    if df is None or df.empty:
        return None
    df = df.sort_values("timestamp").reset_index(drop=True)
    # 去掉推荐日当天（有时 end_time 边界可能包含）
    df = df[df["trade_date"] < pick_date_str].tail(n)
    return df if len(df) >= 5 else None


def calc_metrics(df):
    """对一段日K计算关键指标。"""
    close  = df["close"].values
    volume = df["volume"].values
    n      = len(close)

    ma5  = close[-5:].mean()  if n >= 5  else np.nan
    ma10 = close[-10:].mean() if n >= 10 else np.nan
    ma20 = close[-20:].mean() if n >= 20 else np.nan

    last  = close[-1]
    chg5  = (last - close[-6])  / close[-6]  * 100 if n >= 6  else np.nan
    chg10 = (last - close[-11]) / close[-11] * 100 if n >= 11 else np.nan
    chg20 = (last - close[-21]) / close[-21] * 100 if n >= 21 else np.nan

    vol_avg20 = volume[-20:].mean() if n >= 20 else volume.mean()
    vol_avg5  = volume[-5:].mean()  if n >= 5  else volume.mean()
    vol_ratio = vol_avg5 / vol_avg20 if vol_avg20 > 0 else np.nan   # 近5日量/20日均量

    # 推荐前最后一日量相比20日均量
    last_vol_ratio = volume[-1] / vol_avg20 if vol_avg20 > 0 else np.nan

    # 最近高点位置（20日内最高收盘价）
    hi20  = close[-20:].max() if n >= 20 else close.max()
    near_high = (last / hi20) >= 0.95   # 距20日高点5%以内

    # MA多头排列
    ma_bullish = (not np.isnan(ma5) and not np.isnan(ma10) and not np.isnan(ma20)
                  and ma5 > ma10 > ma20)

    # 价格在MA20之上
    above_ma20 = (not np.isnan(ma20) and last > ma20)

    # 最近是否有缩量回调（近5日量 < 20日均量 0.8）
    pullback_low_vol = (vol_ratio < 0.8) if not np.isnan(vol_ratio) else False

    # 最后一日是否放量（>= 20日均量 1.5 倍）
    last_vol_surge = (last_vol_ratio >= 1.5) if not np.isnan(last_vol_ratio) else False

    return {
        "chg5":            round(chg5,  2),
        "chg10":           round(chg10, 2),
        "chg20":           round(chg20, 2),
        "vol_ratio_5d":    round(vol_ratio, 2),
        "last_vol_ratio":  round(last_vol_ratio, 2),
        "near_20d_high":   near_high,
        "ma_bullish":      ma_bullish,
        "above_ma20":      above_ma20,
        "pullback_low_vol":pullback_low_vol,
        "last_vol_surge":  last_vol_surge,
        "last_close":      round(last, 2),
        "ma5":             round(ma5,  2),
        "ma10":            round(ma10, 2),
        "ma20":            round(ma20, 2),
    }


def main():
    tf = TickFlow.free()
    print(f"\n{'='*70}")
    print("  「为了生活为了家」选股逻辑分析")
    print(f"{'='*70}\n")

    results = []
    errors  = []

    for pick_date, name, symbol in PICKS:
        df = get_kline_before(tf, symbol, pick_date)
        if df is None:
            errors.append((pick_date, name, symbol, "无数据"))
            continue
        m = calc_metrics(df)
        m.update({"date": pick_date, "name": name, "symbol": symbol})
        results.append(m)
        print(f"✓ {pick_date}  {name}({symbol})  "
              f"5日涨跌:{m['chg5']:+.1f}%  10日:{m['chg10']:+.1f}%  "
              f"量比5d:{m['vol_ratio_5d']:.2f}  末日量比:{m['last_vol_ratio']:.2f}  "
              f"MA多头:{m['ma_bullish']}  近高点:{m['near_20d_high']}")

    if errors:
        print(f"\n⚠ 以下记录无数据（可能停牌或代码有误）：")
        for e in errors:
            print(f"  {e[0]}  {e[1]}({e[2]}): {e[3]}")

    if not results:
        print("没有成功拉取到任何数据，请检查网络。")
        return

    rdf = pd.DataFrame(results)

    # ── 统计汇总 ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  汇总统计（共 {n} 条推荐）".format(n=len(results)))
    print(f"{'='*70}")

    n = len(results)

    def pct(col, cond):
        return f"{rdf[col].apply(cond).sum()}/{n} ({rdf[col].apply(cond).mean()*100:.0f}%)"

    print(f"\n【趋势特征（推荐日前）】")
    print(f"  5日涨幅  >0%  : {pct('chg5',  lambda x: x > 0)}")
    print(f"  5日涨幅 >5%   : {pct('chg5',  lambda x: x > 5)}")
    print(f"  10日涨幅 >0%  : {pct('chg10', lambda x: x > 0)}")
    print(f"  10日涨幅 >10% : {pct('chg10', lambda x: x > 10)}")
    print(f"  20日涨幅 >0%  : {pct('chg20', lambda x: x > 0)}")
    print(f"  均线多头排列  : {rdf['ma_bullish'].sum()}/{n} ({rdf['ma_bullish'].mean()*100:.0f}%)")
    print(f"  价格>MA20     : {rdf['above_ma20'].sum()}/{n} ({rdf['above_ma20'].mean()*100:.0f}%)")
    print(f"  处于20日高点区: {rdf['near_20d_high'].sum()}/{n} ({rdf['near_20d_high'].mean()*100:.0f}%)")

    print(f"\n【量能特征（推荐日前）】")
    print(f"  近5日量/20日均量 >1.0 (放量): {pct('vol_ratio_5d', lambda x: x > 1.0)}")
    print(f"  近5日量/20日均量 <0.8 (缩量): {pct('vol_ratio_5d', lambda x: x < 0.8)}")
    print(f"  末日量比 >= 1.5 (末日放量)  : {rdf['last_vol_surge'].sum()}/{n} ({rdf['last_vol_surge'].mean()*100:.0f}%)")
    print(f"  末日量比 均值: {rdf['last_vol_ratio'].mean():.2f}x")

    # ── 涨幅分布 ────────────────────────────────────────────────────────
    print(f"\n【涨幅分布】")
    print(f"  5日涨幅  均值: {rdf['chg5'].mean():+.1f}%  中位数: {rdf['chg5'].median():+.1f}%")
    print(f"  10日涨幅 均值: {rdf['chg10'].mean():+.1f}%  中位数: {rdf['chg10'].median():+.1f}%")
    print(f"  20日涨幅 均值: {rdf['chg20'].mean():+.1f}%  中位数: {rdf['chg20'].median():+.1f}%")

    # ── 典型选股模式识别 ────────────────────────────────────────────────
    print(f"\n【复合模式识别】")

    # 模式1: 强势突破（近5日涨幅>5% + MA多头 + 末日放量）
    m1 = (rdf['chg5'] > 5) & rdf['ma_bullish'] & rdf['last_vol_surge']
    print(f"  ① 强势突破型（5日>5% + MA多头 + 末日放量）: {m1.sum()}/{n} ({m1.mean()*100:.0f}%)")

    # 模式2: 缩量横盘（近5日量/均量<0.9 + 近20日整体上涨）
    m2 = (rdf['vol_ratio_5d'] < 0.9) & (rdf['chg20'] > 5)
    print(f"  ② 强势缩量调整型（5日量比<0.9 + 20日涨>5%）: {m2.sum()}/{n} ({m2.mean()*100:.0f}%)")

    # 模式3: 趋势启动（MA多头 + 价格>MA20 + 末日放量）
    m3 = rdf['ma_bullish'] & rdf['above_ma20'] & rdf['last_vol_surge']
    print(f"  ③ 趋势启动型（MA多头 + 价>MA20 + 末日放量）: {m3.sum()}/{n} ({m3.mean()*100:.0f}%)")

    # 模式4: 高位惯性（处于20日高点 + 5日涨幅>0）
    m4 = rdf['near_20d_high'] & (rdf['chg5'] > 0)
    print(f"  ④ 高位惯性型（近20日高点区 + 5日上涨）     : {m4.sum()}/{n} ({m4.mean()*100:.0f}%)")

    # ── 明细表 ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  明细数据")
    print(f"{'='*70}")
    display_cols = ["date","name","chg5","chg10","chg20",
                    "vol_ratio_5d","last_vol_ratio","ma_bullish","above_ma20","near_20d_high"]
    print(rdf[display_cols].to_string(index=False))

    print(f"\n{'='*70}")
    print("  分析完成")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
