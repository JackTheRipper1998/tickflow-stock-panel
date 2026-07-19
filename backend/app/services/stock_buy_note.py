"""个股买点备注 —— 基于关键价位结构生成「客观数据 + 参考买点 + 理由」。

供策略结果表的「买点备注」列按需批量调用。纯规则、无 LLM、无 IO(价位取自
KlineRepository 内存缓存),毫秒级。输出三段:
  - data:       客观数据(现价 / MA20 及偏离 / 最近支撑 / 上方压力 / 挤压状态)
  - suggestion: 一句话参考买点(回踩位 / 突破位)
  - reason:     买点理由(为何选这个位、当前风险点)

买点规则(追涨语境:标的多为明确上涨趋势):
  1) 偏离 MA20 过大(>18%)     → 追高风险大, 建议等回踩最近支撑
  2) 当日收阴 / 短期转弱          → 观察回踩支撑能否企稳, 不宜追
  3) 处于挤压中                  → 变盘临近, 等放量突破上轨再介入
  4) 贴近新高 / 上方空间小        → 突破型: 放量站上压力位为确认
  5) 健康趋势(默认)            → 回踩近 MA20 的支撑为较优买点, 突破压力可加
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from app.indicators.levels import compute_levels, compute_squeeze

logger = logging.getLogger(__name__)

_SUPPORT_KINDS = ("sr", "pivot", "extreme", "gap")
_RESIST_KINDS = ("sr", "pivot", "extreme", "gap")


def _nearest(levels: dict, side: str, close: float, kinds, n: int = 2):
    """指定方向、距现价最近的 n 个价位 → [(value, label), ...]。"""
    pts = []
    for k in kinds:
        for p in levels.get(k, []):
            if p.get("side") == side and p.get("value"):
                pts.append((abs(p["value"] - close), p["value"], p["label"]))
    pts.sort()
    return [(v, lbl) for _, v, lbl in pts[:n]]


def build_buy_note(repo, symbol: str, days: int = 250) -> dict | None:
    """单只标的的买点备注。数据不足返回 None。"""
    end = date.today()
    start = end - timedelta(days=days * 2)
    try:
        df = repo.get_daily_asset(repo.resolve_asset_type(symbol), symbol, start, end)
    except Exception as e:  # noqa: BLE001
        logger.debug("buy_note %s load failed: %s", symbol, e)
        return None
    if df is None or df.is_empty() or "close" not in df.columns:
        return None

    levels = compute_levels(df)
    squeeze = compute_squeeze(df)
    close = float(df.tail(1)["close"][0])
    last = df.tail(1)
    prev = df.tail(2).head(1)
    ma20 = float(last["ma20"][0]) if "ma20" in df.columns and last["ma20"][0] is not None else None
    chg = None
    if "change_pct" in df.columns and last["change_pct"][0] is not None:
        chg = float(last["change_pct"][0])
    elif prev.height and "close" in df.columns:
        pc = float(prev["close"][0])
        chg = (close / pc - 1) if pc else None

    sup = _nearest(levels, "support", close, _SUPPORT_KINDS, 2)
    res = _nearest(levels, "resistance", close, _RESIST_KINDS, 2)
    bias = (close / ma20 - 1) if ma20 else None

    # —— data 段:客观数据 ——
    dparts = [f"现价 {close:.2f}"]
    if ma20:
        dparts.append(f"MA20 {ma20:.2f}(偏离{bias * 100:+.0f}%)")
    if sup:
        dparts.append("支撑 " + "/".join(f"{v:.2f}" for v, _ in sup))
    if res:
        dparts.append("压力 " + "/".join(f"{v:.2f}" for v, _ in res))
    if squeeze and squeeze.get("on"):
        dparts.append(f"挤压中{squeeze.get('bars')}日")
    data = " · ".join(dparts)

    # —— suggestion + reason 段:规则化参考买点 ——
    sup1 = sup[0] if sup else None
    res1 = res[0] if res else None
    near_high = res1 and ("新高" in res1[1] or "前高" in res1[1])

    if squeeze and squeeze.get("on"):
        suggestion = f"等放量突破 {res1[0]:.2f} 再介入" if res1 else "等挤压向上放量释放再介入"
        reason = f"波动挤压中(已{squeeze.get('bars')}日),方向未明,突破上轨放量方为变盘确认。"
    elif bias is not None and bias > 0.18:
        s = f"回踩 {sup1[0]:.2f}({sup1[1]}) 一带" if sup1 else "回踩企稳后"
        suggestion = f"追高风险大,{s}再考虑"
        reason = f"偏离 MA20 已 {bias * 100:+.0f}%,追高易套;回踩到支撑盈亏比更佳。"
    elif chg is not None and chg < 0:
        s = f"{sup1[0]:.2f}({sup1[1]})" if sup1 else "近端支撑"
        suggestion = f"观察回踩 {s} 能否企稳"
        reason = f"当日收阴 {chg * 100:+.1f}%,短期转弱,不宜追;支撑企稳再择机。"
    elif near_high:
        suggestion = f"放量站上 {res1[0]:.2f}({res1[1]}) 为突破确认"
        reason = f"贴近{res1[1]},突破型形态;成败看量能,回踩 {sup1[0]:.2f} 不破可持。" if sup1 \
            else f"贴近{res1[1]},突破型形态,成败看量能。"
    else:
        s = f"{sup1[0]:.2f}({sup1[1]})" if sup1 else "近 MA20 支撑"
        r = f",突破 {res1[0]:.2f} 放量可加" if res1 else ""
        suggestion = f"回踩 {s} 为较优买点{r}"
        bias_txt = f"偏离 MA20 {bias * 100:+.0f}%,趋势健康;" if bias is not None else ""
        reason = f"{bias_txt}回踩支撑接近 MA20,趋势未破前为顺势买点。"

    return {"data": data, "suggestion": suggestion, "reason": reason}


def build_buy_notes(repo, symbols: list[str], limit: int = 120) -> dict[str, dict]:
    """批量生成买点备注 → {symbol: {data, suggestion, reason}}。超过 limit 只截断。"""
    out: dict[str, dict] = {}
    for sym in symbols[:limit]:
        note = build_buy_note(repo, sym)
        if note:
            out[sym] = note
    return out
