// 看板 · 自选概念实时
// 把"我的自选股所属概念"聚合成卡片网格: 每张卡 = 概念名 + 当前等权平均涨幅
// + 该概念全市场成分股的今日分时线(SVG) + 资金角标(今日成交额 / 放量倍数)。
// 支持 涨幅榜/资金榜 切换 与 开盘啦/同花顺 概念源切换。数据来自后端实时计算,
// 盘中每 45s 轮询一次。
import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { TrendingUp, Wallet, RefreshCw, ChevronDown, ChevronRight, Search, Sparkles, X } from 'lucide-react'
import { api, type ConceptLine } from '@/lib/api'
import { storage } from '@/lib/storage'
import { fmtPct, fmtBigNum, priceColorClass } from '@/lib/format'

type SortMode = 'strength' | 'pct' | 'money'
type Source = 'kpl' | 'ths'
const SOURCE_LABEL: Record<Source, string> = { kpl: '开盘啦', ths: '同花顺' }
// 展示数量档位 (60 = 全部, 后端上限)
const LIMIT_OPTIONS: { label: string; n: number }[] = [
  { label: '12', n: 12 }, { label: '18', n: 18 }, { label: '24', n: 24 }, { label: '全部', n: 60 },
]

interface Props {
  /** 点击展开列表里的自选股 → 打开个股预览(看板传入) */
  onStockClick?: (symbol: string, name: string) => void
}

/** 内联 SVG 分时线: series 为 {t, v(涨幅小数)}。零轴基准 + 按末值红绿着色。 */
function Sparkline({ series, up }: { series: { t: string; v: number }[]; up: boolean }) {
  const W = 168
  const H = 40
  const path = useMemo(() => {
    const vals = series.map(p => p.v).filter(v => Number.isFinite(v))
    if (vals.length < 2) return { d: '', zeroY: null as number | null }
    let lo = Math.min(...vals, 0)
    let hi = Math.max(...vals, 0)
    if (hi === lo) { hi += 1e-4; lo -= 1e-4 }
    const pad = 3
    const x = (i: number) => (i / (series.length - 1)) * W
    const y = (v: number) => H - pad - ((v - lo) / (hi - lo)) * (H - pad * 2)
    const d = series.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(p.v).toFixed(1)}`).join(' ')
    const zeroY = lo <= 0 && hi >= 0 ? y(0) : null
    return { d, zeroY }
  }, [series])

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none" className="block">
      {path.zeroY != null && (
        <line x1="0" x2={W} y1={path.zeroY} y2={path.zeroY} stroke="currentColor" className="text-border" strokeWidth="0.5" strokeDasharray="2 2" />
      )}
      <path d={path.d} fill="none" stroke="currentColor" strokeWidth="1.4" className={up ? 'text-bull' : 'text-bear'} />
    </svg>
  )
}

export function WatchlistConceptIntraday({ onStockClick }: Props) {
  const [sortMode, setSortMode] = useState<SortMode>(() => storage.dashConceptSort.get('strength'))
  const [source, setSource] = useState<Source>(() => storage.watchlistConceptSource.get('kpl'))
  const [displayN, setDisplayN] = useState<number>(() => storage.dashConceptLimit.get(18))
  const [search, setSearch] = useState('')
  const [expanded, setExpanded] = useState<string | null>(null)

  const changeSort = (s: SortMode) => { storage.dashConceptSort.set(s); setSortMode(s) }
  const changeSource = (s: Source) => { storage.watchlistConceptSource.set(s); setSource(s) }
  const changeLimit = (n: number) => { storage.dashConceptLimit.set(n); setDisplayN(n) }

  // 一次性取全部(≤60)概念, 展示按 displayN 截断; 搜索时在全量里过滤 → 搜得到任意概念/自选股
  const query = useQuery({
    queryKey: ['concept-intraday', source, sortMode],
    queryFn: () => api.conceptIntradayLines({ source, sort: sortMode, limit: 60 }),
    refetchInterval: 45_000,
    staleTime: 30_000,
  })

  const items = query.data?.items ?? []
  const q = search.trim().toLowerCase()
  const shown = useMemo(() => {
    if (!q) return items.slice(0, displayN)
    return items.filter(it =>
      it.concept.toLowerCase().includes(q) ||
      (it.watchlist_members ?? []).some(m =>
        m.name.toLowerCase().includes(q) || m.symbol.toLowerCase().includes(q)),
    )
  }, [items, q, displayN])

  return (
    <div className="rounded-lg border border-border bg-surface/40 p-3">
      {/* 头部 */}
      <div className="flex items-center gap-2 flex-wrap mb-2.5">
        <h3 className="text-sm font-medium text-foreground shrink-0">自选概念实时</h3>
        {query.isFetching && <RefreshCw className="h-3 w-3 animate-spin text-muted" />}

        {/* 搜索: 概念名 或 自选股(名称/代码) */}
        <div className="inline-flex items-center gap-1 h-6 px-1.5 rounded bg-elevated text-secondary">
          <Search className="h-3 w-3 shrink-0 text-muted" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="搜概念 / 自选股"
            className="h-5 w-28 bg-transparent text-[11px] outline-none placeholder:text-muted"
          />
          {search && (
            <button onClick={() => setSearch('')} className="text-muted hover:text-foreground shrink-0"><X className="h-3 w-3" /></button>
          )}
        </div>

        <div className="flex-1" />

        {/* 排序切换 */}
        <div className="inline-flex items-center rounded bg-elevated p-0.5">
          <button
            onClick={() => changeSort('strength')}
            title="综合强度: 净流入+放量+涨幅 加权(量价资金共振)"
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] transition-colors ${sortMode === 'strength' ? 'bg-accent/20 text-accent font-medium' : 'text-muted hover:text-foreground'}`}
          >
            <Sparkles className="h-3 w-3" />综合
          </button>
          <button
            onClick={() => changeSort('pct')}
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] transition-colors ${sortMode === 'pct' ? 'bg-accent/20 text-accent font-medium' : 'text-muted hover:text-foreground'}`}
          >
            <TrendingUp className="h-3 w-3" />涨幅
          </button>
          <button
            onClick={() => changeSort('money')}
            title="资金榜: 按等权净流入占比"
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] transition-colors ${sortMode === 'money' ? 'bg-accent/20 text-accent font-medium' : 'text-muted hover:text-foreground'}`}
          >
            <Wallet className="h-3 w-3" />资金
          </button>
        </div>

        {/* 数量档位 */}
        <div className="inline-flex items-center rounded bg-elevated p-0.5">
          {LIMIT_OPTIONS.map(o => (
            <button
              key={o.n}
              onClick={() => changeLimit(o.n)}
              className={`px-1.5 py-0.5 rounded text-[10px] transition-colors ${displayN === o.n ? 'bg-accent/20 text-accent font-medium' : 'text-muted hover:text-foreground'}`}
            >
              {o.label}
            </button>
          ))}
        </div>

        {/* 概念源切换 */}
        <div className="inline-flex items-center rounded bg-elevated p-0.5">
          {(['kpl', 'ths'] as const).map(s => (
            <button
              key={s}
              onClick={() => changeSource(s)}
              className={`px-1.5 py-0.5 rounded text-[10px] transition-colors ${source === s ? 'bg-accent/20 text-accent font-medium' : 'text-muted hover:text-foreground'}`}
              title={s === 'kpl' ? '开盘啦题材' : '同花顺概念'}
            >
              {SOURCE_LABEL[s]}
            </button>
          ))}
        </div>
      </div>

      {/* 内容 */}
      {query.isLoading ? (
        <div className="text-xs text-muted py-6 text-center">加载中…</div>
      ) : items.length === 0 ? (
        <div className="text-xs text-muted py-6 text-center">
          暂无数据 — 需要有自选股, 且当日已有分钟行情与{SOURCE_LABEL[source]}概念数据
        </div>
      ) : shown.length === 0 ? (
        <div className="text-xs text-muted py-6 text-center">没有匹配「{search}」的概念或自选股</div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-2 items-start">
          {shown.map(it => (
            <ConceptCard
              key={it.concept}
              item={it}
              highlight={sortMode}
              expanded={expanded === it.concept || (!!q && (it.watchlist_members ?? []).length > 0)}
              onToggle={() => setExpanded(e => (e === it.concept ? null : it.concept))}
              onStockClick={onStockClick}
              searchTerm={q}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function ConceptCard({ item, highlight, expanded, onToggle, onStockClick, searchTerm }: {
  item: ConceptLine
  highlight: SortMode
  expanded: boolean
  onToggle: () => void
  onStockClick?: (symbol: string, name: string) => void
  searchTerm?: string
}) {
  const up = (item.avg_pct ?? 0) >= 0
  const vr = item.vol_ratio
  const ir = item.inflow_ratio
  // 放量色: >=1.5 强, >=1 温和, <1 缩量
  const vrClass = vr == null ? 'text-muted' : vr >= 1.5 ? 'text-warning' : vr >= 1 ? 'text-secondary' : 'text-muted'
  const moneyHi = highlight === 'money'
  const wm = item.watchlist_members ?? []
  return (
    <div
      onClick={onToggle}
      className={`rounded-md border bg-surface/60 p-2 flex flex-col gap-1.5 cursor-pointer transition-colors ${expanded ? 'border-accent/50' : 'border-border/60 hover:border-accent/30'}`}
    >
      {/* 顶行: 概念名 + 当前均涨 */}
      <div className="flex items-baseline justify-between gap-1">
        <span className="text-xs font-medium text-foreground truncate" title={`${item.concept} · ${item.member_count}只成分 · 今日成交额 ${fmtBigNum(item.amount_today)}`}>
          {item.concept}
        </span>
        <span className={`text-sm font-semibold tabular-nums shrink-0 ${priceColorClass(item.avg_pct)}`}>
          {fmtPct(item.avg_pct, 2)}
        </span>
      </div>

      {/* 分时线 */}
      <Sparkline series={item.series} up={up} />

      {/* 底部资金角标: 放量倍数 · 净流入占比 */}
      <div className="flex items-center justify-between text-[10px] leading-none gap-1">
        <span className="text-muted tabular-nums shrink-0">{item.member_count}只</span>
        <div className="flex items-center gap-1.5 tabular-nums">
          <span className={vrClass} title="放量倍数: 今日成交额 vs 近5日同时段(≥1.5×强放量)">
            放{vr != null ? `${vr.toFixed(2)}×` : '—'}
          </span>
          <span
            className={`font-medium ${ir == null ? 'text-muted' : priceColorClass(ir)} ${moneyHi ? 'ring-1 ring-inset ring-current/30 rounded px-1 -mx-0.5' : ''}`}
            title="资金净流入占比(等权): 每只成分股各自由分时买卖压力估算净流入占比, 再等权平均(已过滤低成交票), 不受成分股数量或单只大市值龙头主导。正=多数股票资金净买入(主力进场), 负=净卖出"
          >
            净{ir != null ? fmtPct(ir, 1) : '—'}
          </span>
        </div>
      </div>

      {/* 命中的自选股: 点击展开 */}
      {wm.length > 0 && (
        <div className="border-t border-border/50 pt-1 -mb-0.5">
          <div className="flex items-center gap-1 text-[10px] text-accent">
            {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            含自选 {wm.length} 只
          </div>
          {expanded && (
            <div className="mt-1 space-y-0.5">
              {wm.map(s => {
                const hit = !!searchTerm && (s.name.toLowerCase().includes(searchTerm) || s.symbol.toLowerCase().includes(searchTerm))
                return (
                  <button
                    key={s.symbol}
                    onClick={(e) => { e.stopPropagation(); onStockClick?.(s.symbol, s.name) }}
                    className={`flex w-full items-center justify-between gap-1 rounded px-1 py-0.5 text-[11px] transition-colors ${hit ? 'bg-accent/15 ring-1 ring-accent/40' : 'hover:bg-elevated'}`}
                  >
                    <span className={`truncate ${hit ? 'text-accent font-medium' : 'text-foreground'}`}>{s.name}</span>
                    <span className={`tabular-nums shrink-0 ${priceColorClass(s.pct)}`}>{fmtPct(s.pct, 1)}</span>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
