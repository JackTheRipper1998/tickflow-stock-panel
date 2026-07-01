"""前高再攻 — 经历一轮完整下跌趋势后上涨，当前接近前高（下跌趋势起点）"""
import polars as pl

META = {
    "id": "near_prev_high",
    "name": "前高再攻",
    "description": "经历一轮完整下跌趋势后上涨，当前价格接近前高（下跌起点）",
    "tags": ["前高", "回踩", "再攻"],
    "params": [
        {"id": "gap_pct", "label": "距前高最大比例", "type": "float",
         "default": 0.15, "min": 0.02, "max": 0.20, "step": 0.01},
        {"id": "min_drop_pct", "label": "最小下跌幅度", "type": "float",
         "default": 0.10, "min": 0.05, "max": 0.40, "step": 0.01},
        {"id": "min_down_days", "label": "最少下跌天数", "type": "int",
         "default": 5, "min": 3, "max": 30, "step": 1},
        {"id": "min_up_days", "label": "最少上涨天数", "type": "int",
         "default": 5, "min": 3, "max": 30, "step": 1},
        {"id": "max_bounce_ratio", "label": "下跌段最大反弹比", "type": "float",
         "default": 0.65, "min": 0.10, "max": 0.80, "step": 0.05},
        {"id": "max_down_up_ratio", "label": "下跌段最多上涨日比例", "type": "float",
         "default": 0.40, "min": 0.20, "max": 0.55, "step": 0.05},
    ],
    "scoring": {"momentum_20d": 0.4, "momentum_10d": 0.3, "vol_ratio_5d": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

ENTRY_SIGNALS = []
EXIT_SIGNALS = ["signal_ma20_breakdown"]
STOP_LOSS = -0.08
MAX_HOLD_DAYS = 20
ALERTS = []
LOOKBACK_DAYS = 200


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    gap_pct = float(params.get("gap_pct", 0.15))
    min_drop_pct = float(params.get("min_drop_pct", 0.10))
    min_down_days = int(params.get("min_down_days", 5))
    min_up_days = int(params.get("min_up_days", 5))
    max_bounce_ratio = float(params.get("max_bounce_ratio", 0.65))
    max_down_up_ratio = float(params.get("max_down_up_ratio", 0.40))

    df = df.sort(["symbol", "date"])
    latest_date = df["date"].max()

    def _check(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort("date")
        closes = group["close"].to_numpy()
        n = len(closes)
        if n < 15:
            return group.head(0)

        # 前高：不含最后一天的最高收盘（= 下跌趋势起点）
        peak_idx = int(closes[:-1].argmax())
        peak_price = float(closes[peak_idx])

        # 前高之后的低谷
        after_peak = closes[peak_idx:]
        trough_rel_idx = int(after_peak.argmin())
        trough_price = float(after_peak[trough_rel_idx])
        trough_abs_idx = peak_idx + trough_rel_idx

        current_price = float(closes[-1])
        total_drop = peak_price - trough_price
        drop_pct = total_drop / peak_price
        gap_to_high = (peak_price - current_price) / peak_price

        downtrend_days = trough_abs_idx - peak_idx
        uptrend_days = (n - 1) - trough_abs_idx

        # 下跌段质量检查：确保是一轮连贯下跌（两个维度）
        bounce_ratio = 0.0       # 最大中间反弹占总跌幅的比例
        down_up_ratio = 0.0     # 下跌段中上涨日的占比
        if downtrend_days > 1 and total_drop > 0:
            mid_closes = closes[peak_idx + 1:trough_abs_idx]
            if len(mid_closes) > 0:
                mid_high = float(mid_closes.max())
                bounce_ratio = (mid_high - trough_price) / total_drop
            down_seg = closes[peak_idx:trough_abs_idx + 1]
            down_up_days_count = int((down_seg[1:] > down_seg[:-1]).sum())
            down_up_ratio = down_up_days_count / (len(down_seg) - 1)

        # 上涨段质量：多数交易日收盘上涨（趋势性上涨，非震荡）
        up_segment = closes[trough_abs_idx:]
        up_day_ratio = 0.0
        if len(up_segment) > 1:
            up_days_count = int((up_segment[1:] > up_segment[:-1]).sum())
            up_day_ratio = up_days_count / (len(up_segment) - 1)

        ok = (
            trough_abs_idx > peak_idx               # 低谷在前高之后
            and trough_abs_idx < n - 1              # 低谷不是最后一天
            and drop_pct >= min_drop_pct            # 下跌幅度足够（真实下跌趋势）
            and downtrend_days >= min_down_days     # 下跌持续天数足够
            and uptrend_days >= min_up_days         # 上涨已持续足够天数
            and bounce_ratio <= max_bounce_ratio    # 下跌段无大幅反弹（一轮下跌）
            and down_up_ratio <= max_down_up_ratio  # 下跌段多数为下跌日（趋势性下跌）
            and up_day_ratio >= 0.5                 # 上涨段多数为上涨日（趋势性）
            and current_price > trough_price        # 当前高于低谷
            and current_price < peak_price          # 尚未突破前高
            and 0 < gap_to_high <= gap_pct          # 当前价已回到接近前高位置
        )

        if ok:
            return group.filter(pl.col("date") == latest_date)
        return group.head(0)

    return df.group_by("symbol").map_groups(_check)
