"""行情状态 / SSE 推送 API。

盘中选股相关端点已迁移至策略页面，此处仅保留全局行情基础设施。
SSE 推送三种事件 (使用标准 SSE event 字段):
  - quotes_updated: 行情数据刷新，前端 invalidate 对应 query
  - strategy_alert: 策略监控/告警触发，前端弹通知
  - depth_updated: 五档盘口修正完成，前端刷新连板梯队/看板封单数据
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, Query, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter(prefix="/api/intraday", tags=["quotes"])


def _get_quote_service(request: Request):
    """获取全局 QuoteService。"""
    return getattr(request.app.state, "quote_service", None)


def _fallback_index_quotes_from_daily(request: Request, symbols: list[str] | None = None) -> list[dict]:
    """实时指数缓存为空时，从本地指数日 K 取最近收盘价作为兜底。"""
    repo = getattr(request.app.state, "repo", None)
    if not repo:
        return []

    params: list[str] = []
    symbol_filter = ""
    if symbols:
        placeholders = ", ".join("?" for _ in symbols)
        symbol_filter = f"WHERE symbol IN ({placeholders})"
        params.extend(symbols)

    try:
        rows = repo.execute_all(
            f"""
            WITH ranked AS (
                SELECT symbol, date, close,
                       row_number() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
                FROM kline_index_daily
                {symbol_filter}
            ), latest AS (
                SELECT symbol,
                       max(CASE WHEN rn = 1 THEN date END) AS date,
                       max(CASE WHEN rn = 1 THEN close END) AS last_price,
                       max(CASE WHEN rn = 2 THEN close END) AS prev_close
                FROM ranked
                WHERE rn <= 2
                GROUP BY symbol
            )
            SELECT latest.symbol, latest.date, latest.last_price, latest.prev_close
            FROM latest
            ORDER BY latest.symbol
            """,
            params,
        )
    except Exception:  # noqa: BLE001
        return []

    out: list[dict] = []
    for symbol, dt, last_price, prev_close in rows:
        change_amount = None
        change_pct = None
        if last_price is not None and prev_close not in (None, 0):
            change_amount = float(last_price) - float(prev_close)
            change_pct = change_amount / float(prev_close) * 100
        out.append({
            "symbol": symbol,
            "name": None,
            "date": str(dt) if dt else None,
            "last_price": float(last_price) if last_price is not None else None,
            "close": float(last_price) if last_price is not None else None,
            "prev_close": float(prev_close) if prev_close is not None else None,
            "change_amount": change_amount,
            "change_pct": change_pct,
            "source": "index_daily",
        })
    return out


@router.get("/status")
def status(request: Request):
    """行情状态 (来自全局 QuoteService)。"""
    qs = _get_quote_service(request)
    if qs:
        return qs.status()
    return {"enabled": False, "running": False, "symbol_count": 0, "index_symbol_count": 0,
            "quote_age_ms": None, "is_trading_hours": False, "last_fetch_ms": None}


@router.get("/indices")
def index_quotes(
    request: Request,
    symbols: str | None = Query(None, description="逗号分隔的指数 symbol 列表"),
):
    """返回实时指数行情缓存，不触发 TickFlow 请求。"""
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
    qs = _get_quote_service(request)
    if not qs:
        rows = _fallback_index_quotes_from_daily(request, symbol_list)
        return {"rows": rows, "count": len(rows), "source": "index_daily"}
    df = qs.get_index_quotes(symbol_list)
    rows = df.to_dicts() if not df.is_empty() else []
    if not rows:
        rows = _fallback_index_quotes_from_daily(request, symbol_list)
        return {"rows": rows, "count": len(rows), "source": "index_daily"}
    return {"rows": rows, "count": len(rows), "source": "realtime"}


@router.get("/stream")
async def quote_stream(request: Request):
    """SSE 端点: 行情更新 + 告警推送 + 五档修正 + 复盘进度。

    使用 sse-starlette EventSourceResponse:
    - 标准 SSE event 字段，前端按 event name 监听
    - 内置断线检测，客户端断开立即终止 generator
    - 内置 ping 心跳，保持连接活跃

    每个连接注册一个独立订阅者 (QuoteSubscriber: 独立事件 + 独立队列),
    事件由 QuoteService 广播 — 多客户端 (多标签页/设备) 各自收到全量事件。
    此前四通道共用服务级 Event + pop 取走语义, 告警只会被先醒的连接消费。
    """
    qs = _get_quote_service(request)

    async def event_generator():
        if qs is None:
            # 无行情服务: 保持连接 (EventSourceResponse 自带 ping), 不推事件
            while True:
                await asyncio.sleep(30)

        sub = qs.subscribe()
        try:
            while True:
                # 等待任一通道有新信号 (5s 超时保持循环, 便于断线时尽快退出)
                await asyncio.to_thread(sub.wait, 5.0)
                data = sub.pop()

                # 告警 (分片推送, 避免单条 SSE 过大)
                alerts = data["alerts"]
                for chunk_start in range(0, len(alerts), 20):
                    chunk = alerts[chunk_start:chunk_start + 20]
                    yield {
                        "event": "strategy_alert",
                        "data": json.dumps({
                            "ts": int(time.time() * 1000),
                            "alerts": chunk,
                        }, ensure_ascii=False),
                    }

                # 复盘进度 (定时复盘流式生成时) — 前端 reviewStore 直接消费
                # 事件已是 recap_market_stream 产出的 JSON 字符串, 逐条转发
                for evt_json in data["reviews"]:
                    yield {
                        "event": "review_progress",
                        "data": evt_json,
                    }

                # 行情更新
                if data["quote_updated"]:
                    yield {
                        "event": "quotes_updated",
                        "data": json.dumps({
                            "ts": int(time.time() * 1000),
                            "symbol_count": qs._symbol_count,
                        }),
                    }

                # 五档修正完成 — 前端刷新连板梯队封单数据
                if data["depth_updated"]:
                    yield {
                        "event": "depth_updated",
                        "data": json.dumps({
                            "ts": int(time.time() * 1000),
                        }),
                    }
        finally:
            qs.unsubscribe(sub)

    return EventSourceResponse(event_generator())


@router.post("/refresh")
def refresh_quotes(request: Request):
    """手动刷新一次行情数据。"""
    qs = _get_quote_service(request)
    if qs:
        return qs.refresh()
    return {"error": "QuoteService not available"}


# ── 自选概念实时分时 ───────────────────────────────────────────────────────
# 泛概念黑名单 (与前端 WatchlistConceptPanel.isGenericConcept 保持一致)
_GENERIC_CONCEPTS = {
    "融资融券", "转融券标的", "融资标的", "融券标的",
    "新股与次新股", "注册制次新股", "次新股",
    "标普道琼斯A股", "富时罗素概念", "富时罗素概念股", "MSCI中国", "MSCI概念",
    "深股通", "沪股通", "陆股通", "B股概念", "AH股", "GDR", "H股",
    "央视50", "破净股", "微盘股", "低价股", "破净整理", "高股息精选",
}
_CONCEPT_SOURCE_TABLE = {"kpl": "ext_kpl_theme", "ths": "ext_gn_ths"}


def _is_generic_concept(c: str) -> bool:
    return c in _GENERIC_CONCEPTS or c.endswith(("成份股", "成分股", "样本股", "标的"))


@router.get("/concept-lines")
def concept_intraday_lines(
    request: Request,
    source: str = Query("kpl", description="概念数据源: kpl=开盘啦 | ths=同花顺"),
    sort: str = Query("strength", description="排序: strength=综合强度 | pct=涨幅榜 | money=资金榜(净流入)"),
    limit: int = Query(24, ge=1, le=60, description="返回前 N 个概念"),
):
    """自选股所属概念的今日实时分时: 每个概念取其全市场成分股, 算等权平均涨幅的
    分时序列 + 当前均涨 + 今日成交额 + 放量倍数, 供看板"自选概念实时"卡片网格。"""
    import glob

    import polars as pl

    repo = getattr(request.app.state, "repo", None)
    if repo is None:
        return {"as_of": None, "trading": False, "items": []}
    data_dir = repo.store.data_dir

    # 1. 自选
    wl_path = data_dir / "user_data" / "watchlist.parquet"
    if not wl_path.exists():
        return {"as_of": None, "trading": False, "items": []}
    try:
        wl = pl.read_parquet(wl_path)["symbol"].to_list()
    except Exception:  # noqa: BLE001
        return {"as_of": None, "trading": False, "items": []}

    # 2. 概念表 → symbol→概念 / 概念→全市场成分
    table = _CONCEPT_SOURCE_TABLE.get(source, "ext_kpl_theme")
    ext_path = data_dir / "ext_data" / table / "part.parquet"
    if not ext_path.exists():
        return {"as_of": None, "trading": False, "items": [], "reason": f"{table} 未生成"}
    ext = pl.read_parquet(ext_path)
    sym2con: dict[str, list[str]] = {}
    con2mem: dict[str, list[str]] = {}
    sym2name: dict[str, str] = {}
    name_col = "股票简称" if "股票简称" in ext.columns else None
    sel = ["symbol", "所属概念"] + ([name_col] if name_col else [])
    for row in ext.select(sel).iter_rows():
        s, raw = row[0], row[1]
        if name_col and row[2]:
            sym2name[s] = row[2]
        cs = [x.strip() for x in (raw or "").split(";") if x.strip()]
        sym2con[s] = cs
        for c in cs:
            con2mem.setdefault(c, []).append(s)

    # 3. 自选带出的概念 (去重 + 去泛概念)
    wl_concepts: list[str] = []
    seen: set[str] = set()
    for s in wl:
        for c in sym2con.get(s, []):
            if c not in seen and not _is_generic_concept(c):
                seen.add(c)
                wl_concepts.append(c)
    if not wl_concepts:
        return {"as_of": None, "trading": False, "items": []}

    # 4. 今日分钟 (全市场) + 昨收
    mdirs = sorted(glob.glob(str(data_dir / "kline_minute" / "date=*")))
    if not mdirs:
        return {"as_of": None, "trading": False, "items": []}
    today = mdirs[-1].split("date=")[-1]
    try:
        mins = pl.read_parquet(f"{mdirs[-1]}/part.parquet")
    except Exception:  # noqa: BLE001
        return {"as_of": today, "trading": False, "items": []}

    members: set[str] = set()
    for c in wl_concepts:
        members.update(con2mem.get(c, []))

    ddirs = sorted(glob.glob(str(data_dir / "kline_daily" / "date=*")))
    prevdays = [d for d in ddirs if d.split("date=")[-1] < today]
    if not prevdays:
        return {"as_of": today, "trading": False, "items": []}
    daily_prev = (
        pl.read_parquet(f"{prevdays[-1]}/part.parquet")
        .select(["symbol", "close"]).rename({"close": "prev_close"})
    )

    # 5. 分钟涨幅
    m = (
        mins.filter(pl.col("symbol").is_in(list(members)))
        .join(daily_prev, on="symbol", how="inner")
        .with_columns(((pl.col("close") / pl.col("prev_close")) - 1.0).alias("chg"))
    )
    if m.is_empty():
        return {"as_of": today, "trading": False, "items": []}
    # 分钟资金流方向 (买卖压力 MFM ∈ [-1,1]): 收盘越靠近最高→买盘主导, 越靠近最低→卖盘主导。
    # mf = MFM × 该分钟成交额 → 全天累加得净流入额 (通达信/东财"分时资金流"近似)。
    m = m.with_columns(
        pl.when(pl.col("high") > pl.col("low"))
        .then(((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close")))
              / (pl.col("high") - pl.col("low")))
        .otherwise(0.0)
        .alias("_mfm")
    ).with_columns((pl.col("_mfm") * pl.col("amount")).alias("mf"))
    n_bars = m["datetime"].n_unique()
    elapsed_frac = min(1.0, max(n_bars, 1) / 240.0)
    last_dt = m["datetime"].max()

    # 6. concept×symbol 展开
    long_rows = [(c, s) for c in wl_concepts for s in con2mem.get(c, [])]
    mem_df = pl.DataFrame(long_rows, schema=["concept", "symbol"], orient="row")
    joined = mem_df.join(
        m.select(["symbol", "datetime", "chg", "amount", "mf"]), on="symbol", how="inner"
    )

    # 分时序列 (每分钟等权平均涨幅)
    series = (
        joined.group_by(["concept", "datetime"])
        .agg(pl.col("chg").mean().alias("v"))
        .sort(["concept", "datetime"])
    )
    # 当前值 + 成分数 + 今日成交额
    cur = (
        joined.filter(pl.col("datetime") == last_dt)
        .group_by("concept")
        .agg(
            pl.col("chg").mean().alias("avg_pct"),
            pl.col("symbol").n_unique().alias("member_count"),
        )
    )
    amt = joined.group_by("concept").agg(pl.col("amount").sum().alias("amount_today"))

    # 资金净流入占比 (等权口径, 与平均涨幅一致, 避免单只大市值龙头主导):
    #   1) 每只股票各自净流入占比 = Σ(分钟买卖压力×成交额)/Σ成交额 ∈ [-1,1] (本身 size 无关)
    #   2) 过滤当日成交额过小的票 (< 1000万), 避免低流动性个股用 MFM 制造噪声
    #   3) 概念内成分股 *等权* 平均 → 每只票权重相同, 反映"多数股票在被买还是被卖"
    _AMT_FLOOR = 1e7
    per_stock = (
        m.group_by("symbol")
        .agg(pl.col("mf").sum().alias("_net"), pl.col("amount").sum().alias("_amt"))
        .filter(pl.col("_amt") >= _AMT_FLOOR)
        .with_columns((pl.col("_net") / pl.col("_amt")).alias("s_inflow"))
    )
    mem_flow = (
        mem_df.join(per_stock.select(["symbol", "s_inflow"]), on="symbol", how="inner")
        .group_by("concept")
        .agg(pl.col("s_inflow").mean().alias("inflow_ratio"))
    )
    inflow_map = dict(zip(mem_flow["concept"].to_list(), mem_flow["inflow_ratio"].to_list()))

    # 放量倍数: 今日成交额 / (近5日概念日均成交额 × 已过时段占比)
    vol_ratio_map: dict[str, float] = {}
    last5 = prevdays[-5:]
    if last5:
        try:
            d5 = pl.concat([
                pl.read_parquet(f"{d}/part.parquet").select(["symbol", "amount"])
                for d in last5
            ])
            d5j = mem_df.join(d5, on="symbol", how="inner")
            avg5 = (
                d5j.group_by("concept").agg(pl.col("amount").sum().alias("s"))
                .with_columns((pl.col("s") / len(last5)).alias("avg5"))
            )
            avg5_map = dict(zip(avg5["concept"].to_list(), avg5["avg5"].to_list()))
            amt_map = dict(zip(amt["concept"].to_list(), amt["amount_today"].to_list()))
            for c in wl_concepts:
                a5 = avg5_map.get(c)
                at = amt_map.get(c)
                if a5 and a5 > 0 and at is not None and elapsed_frac > 0:
                    vol_ratio_map[c] = at / (a5 * elapsed_frac)
        except Exception:  # noqa: BLE001
            pass

    stats = cur.join(amt, on="concept", how="left")
    rows = stats.to_dicts()
    for r in rows:
        r["vol_ratio"] = vol_ratio_map.get(r["concept"])
        r["inflow_ratio"] = inflow_map.get(r["concept"])

    # 综合强度 (量价资金共振): 净流入/放量/涨幅 各自在概念集内的百分位排名加权。
    def _prank(key: str) -> list[float]:
        vals = [(r.get(key) if r.get(key) is not None else -1e18) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: vals[i])
        n = len(rows)
        out = [0.0] * n
        for pos, i in enumerate(order):
            out[i] = pos / (n - 1) if n > 1 else 0.5
        return out
    r_in, r_vol, r_pct = _prank("inflow_ratio"), _prank("vol_ratio"), _prank("avg_pct")
    for i, r in enumerate(rows):
        r["strength"] = 0.40 * r_in[i] + 0.35 * r_vol[i] + 0.25 * r_pct[i]

    # 排序 + 取 top-N
    if sort == "money":  # 资金榜: 净流入占比(反映主力进出方向, 与成分股数量无关)
        rows.sort(key=lambda r: (r.get("inflow_ratio") if r.get("inflow_ratio") is not None else -1e9), reverse=True)
    elif sort == "pct":  # 涨幅榜
        rows.sort(key=lambda r: (r.get("avg_pct") if r.get("avg_pct") is not None else -1e9), reverse=True)
    else:  # strength: 综合强度(默认)
        rows.sort(key=lambda r: r.get("strength", 0.0), reverse=True)
    top = rows[:limit]
    top_names = {r["concept"] for r in top}

    # 只为 top-N 附分时序列 (转北京时间 HH:MM)
    ser_top = (
        series.filter(pl.col("concept").is_in(list(top_names)))
        .with_columns((pl.col("datetime") + pl.duration(hours=8)).dt.strftime("%H:%M").alias("t"))
    )
    ser_map: dict[str, list[dict]] = {}
    for c, t, v in ser_top.select(["concept", "t", "v"]).iter_rows():
        ser_map.setdefault(c, []).append({"t": t, "v": v})
    # 降采样: 迷你分时线不需要全部 240 点, 概念多时控制包体
    for c, lst in ser_map.items():
        if len(lst) > 140:
            step = (len(lst) + 119) // 120
            ser_map[c] = lst[::step]

    # 每个概念里"命中的自选股"(带当前涨幅), 供前端点开卡片查看
    wl_set = set(wl)
    cur_chg = dict(
        m.filter(pl.col("datetime") == last_dt).select(["symbol", "chg"]).iter_rows()
    )

    items = []
    for r in top:
        c = r["concept"]
        wl_members = sorted(
            [
                {"symbol": s, "name": sym2name.get(s, s), "pct": cur_chg.get(s)}
                for s in con2mem.get(c, []) if s in wl_set
            ],
            key=lambda x: (x["pct"] if x["pct"] is not None else -1e9),
            reverse=True,
        )
        items.append({
            "concept": c,
            "avg_pct": r.get("avg_pct"),
            "member_count": r.get("member_count"),
            "amount_today": r.get("amount_today"),
            "vol_ratio": r.get("vol_ratio"),
            "inflow_ratio": r.get("inflow_ratio"),
            "strength": r.get("strength"),
            "watchlist_members": wl_members,
            "series": ser_map.get(c, []),
        })

    return {"as_of": today, "trading": elapsed_frac < 1.0, "items": items, "source": source, "sort": sort}
