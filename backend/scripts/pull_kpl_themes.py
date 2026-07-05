"""开盘啦(龙虎VIP)题材库拉取 → 生成 ext_kpl_theme 扩展数据预设。

背景
----
开盘啦 App 的题材库(c=Theme, a=InfoGet)数据质量/时效优于同花顺概念,但接口是
登录态(需 Token/UserID/DeviceID),且没有公开的"题材列表"接口 —— 题材 ID 稀疏
分布在 1~350 区间。本脚本:

  1. 枚举 ID(默认 1~2000)找出所有有效题材;
  2. 逐个 InfoGet 抓成分股(StockList: StockID / prod_name);
  3. 反转成"股票→题材"并写入 data/ext_data/ext_kpl_theme/{config.json,part.parquet},
     列结构与 ext_gn_ths(同花顺概念)完全一致, 前端概念分析页 / 自选概念透视面板
     可直接切换使用。

Token 获取
----------
用抓包工具(Fiddler/Charles)对 App 抓一条 host=applhb.longhuvip.com 的
POST /w1/api/index.php 请求, 从 body 里取 Token / UserID / DeviceID。Token 会过期,
过期后重抓一次即可。

用法
----
    python backend/scripts/pull_kpl_themes.py \
        --token <TOKEN> --user-id <UID> --device-id <DID> \
        [--max-id 2000] [--workers 6]

只自用、低频运行, 勿高频轮询。
"""
from __future__ import annotations

import argparse
import gzip
import json
import queue
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import polars as pl

BASE = "https://applhb.longhuvip.com/w1/api/index.php"
UA = "Dalvik/2.1.0 (Linux; U; Android 12; NTH-AN00 Build/096cd29.0)"
ROOT = Path(__file__).resolve().parents[2]  # tickflow-stock-panel/


def make_caller(token: str, user_id: str, device_id: str):
    auth = dict(
        PhoneOSNew="1", DeviceID=device_id, VerSion="5.23.0.4",
        Token=token, apiv="w44", UserID=user_id,
    )

    def call(a: str, c: str, **kw):
        data = dict(auth); data.update(a=a, c=c); data.update(kw)
        body = urllib.parse.urlencode(data).encode()
        req = urllib.request.Request(
            BASE, data=body,
            headers={
                "User-Agent": UA,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Accept-Encoding": "gzip",
            },
        )
        r = urllib.request.urlopen(req, timeout=20)
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))

    return call


def _run_pool(items, fn, workers: int, delay: float, label: str):
    q: queue.Queue = queue.Queue()
    for it in items:
        q.put(it)
    lock = threading.Lock(); done = [0]; total = len(items)

    def worker():
        while True:
            try:
                it = q.get_nowait()
            except queue.Empty:
                return
            try:
                fn(it)
            except Exception:
                pass
            with lock:
                done[0] += 1
                if done[0] % 50 == 0:
                    print(f"  {label}: {done[0]}/{total}", flush=True)
            time.sleep(delay)

    ts = [threading.Thread(target=worker) for _ in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--device-id", required=True)
    ap.add_argument("--max-id", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--delay", type=float, default=0.15)
    args = ap.parse_args()

    call = make_caller(args.token, args.user_id, args.device_id)

    # 连通性自检
    probe = call("InfoGet", "Theme", ID="84")
    if not probe.get("Name"):
        print("[!] 自检失败: ID=84 返回空 — Token 可能已过期或参数有误", file=sys.stderr)
        return 2
    print(f"[✓] Token 有效 (ID=84 → {probe.get('Name')})")

    # 1) 枚举有效题材 ID
    print(f"[1/3] 枚举题材 ID 1~{args.max_id} ...")
    valid: dict[int, str] = {}
    vlock = threading.Lock()

    def enum_one(ID: int):
        d = call("InfoGet", "Theme", ID=str(ID))
        n = d.get("Name")
        if n:
            with vlock:
                valid[ID] = n

    _run_pool(list(range(1, args.max_id + 1)), enum_one, args.workers, args.delay, "枚举")
    print(f"    有效题材: {len(valid)}")

    # 2) 抓详情(枚举其实已取到 StockList, 但为清晰起见二次拉取有效 ID)
    print(f"[2/3] 抓取 {len(valid)} 个题材的成分股 ...")
    stock_themes: dict[str, list[str]] = defaultdict(list)
    stock_name: dict[str, str] = {}
    slock = threading.Lock()

    def detail_one(ID: int):
        d = call("InfoGet", "Theme", ID=str(ID))
        name = d.get("Name")
        if not name:
            return
        for s in (d.get("StockList") or []):
            code = str(s.get("StockID", "")).strip()
            if not code:
                continue
            pn = (s.get("prod_name") or "").strip()
            with slock:
                stock_themes[code].append(name)
                if pn and code not in stock_name:
                    stock_name[code] = pn

    _run_pool(list(valid), detail_one, args.workers, args.delay, "详情")
    print(f"    覆盖股票: {len(stock_themes)}")

    # 3) 反转 → 写 ext_kpl_theme 预设
    print("[3/3] 生成 ext_kpl_theme 预设 ...")
    inst = pl.read_parquet(ROOT / "data/instruments/instruments.parquet")
    name_col = "name" if "name" in inst.columns else None
    code2sym = dict(zip(inst["code"].to_list(), inst["symbol"].to_list()))
    code2name = dict(zip(inst["code"].to_list(), inst[name_col].to_list())) if name_col else {}

    def to_symbol(code: str) -> str:
        s = code2sym.get(code)
        if s:
            return s
        if len(code) == 6 and code.isdigit():
            if code[0] == "6":
                return code + ".SH"
            if code[0] in ("8", "4") or code[:2] == "92":
                return code + ".BJ"
            return code + ".SZ"
        return code

    rows = []
    for code, themes in stock_themes.items():
        seen: set[str] = set(); uniq: list[str] = []
        for t in themes:
            if t and t not in seen:
                seen.add(t); uniq.append(t)
        sym = to_symbol(code)
        rows.append({
            "股票代码": sym,
            "股票简称": code2name.get(code) or stock_name.get(code) or "",
            "所属概念": ";".join(uniq),
            "symbol": sym,
            "code": code,
        })

    df = pl.DataFrame(rows)
    outdir = ROOT / "data/ext_data/ext_kpl_theme"
    outdir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(outdir / "part.parquet")

    now = datetime.now().isoformat()
    cfg = {
        "id": "ext_kpl_theme",
        "label": "开盘啦题材",
        "mode": "snapshot",
        "fields": [
            {"name": "symbol", "dtype": "string", "label": "标的代码"},
            {"name": "code", "dtype": "string", "label": "代码"},
            {"name": "股票代码", "dtype": "string", "label": "股票代码"},
            {"name": "股票简称", "dtype": "string", "label": "股票简称"},
            {"name": "所属概念", "dtype": "string", "label": "所属概念"},
        ],
        "description": "开盘啦题材库分类 (来自开盘啦App题材库, 用 pull_kpl_themes.py 手动更新)",
        "symbol_map": {"type": "mapped", "col": "股票代码"},
        "code_map": {"type": "computed", "from": "symbol", "method": "strip_exchange"},
        "created_at": now,
        "updated_at": now,
    }
    (outdir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[✓] 完成: {len(df)} 只股票 / {len(valid)} 个题材 → {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
