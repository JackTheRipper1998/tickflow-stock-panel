// 自选概念透视 — 把"股票→概念"反过来聚合成"概念→我的哪些股票",
// 折叠面板形态嵌在自选页筛选栏下方。
//
// 数据来源:
//   - 概念表: ext_gn_ths(同花顺概念, 全市场快照), 与概念分析页共用同一
//     react-query 缓存(相同 queryKey), 页面间免重复请求。
//   - 自选实时涨跌: 直接用自选页已有的 enriched rows(rt_pct/change_pct)。
//   - 市场热度: /api/screener/market-snapshot, 仅在面板展开时请求 —
//     计算每个概念全市场平均涨幅与排名, 对比"我的票 vs 整个概念"。
//
// 泛概念降噪: 融资融券/次新股/成份股类概念默认过滤(否则聚合第一名永远是
// 融资融券), 面板内可切换显示。
import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ChevronDown, ChevronRight, Download, RefreshCw, X } from 'lucide-react'
import { api, type MarketSnapshotRow } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { storage } from '@/lib/storage'
import { fmtPct, priceColorClass } from '@/lib/format'

// 概念数据源: 开盘啦题材(时效/质量更好) 或 同花顺概念。两者都是全市场快照,
// 列结构一致(所属概念 = 分号拼接), 切换只换底层 ext 表。
export type ConceptSource = 'kpl' | 'ths'
export const CONCEPT_SOURCE_TABLE: Record<ConceptSource, string> = {
  kpl: 'ext_kpl_theme',
  ths: 'ext_gn_ths',
}
export const CONCEPT_SOURCE_LABEL: Record<ConceptSource, string> = {
  kpl: '开盘啦',
  ths: '同花顺',
}
// 与概念分析页 PAGE_LIMIT 保持一致 → queryKey 相同, 两页共享缓存
const CONCEPT_LIMIT = 12000
const CONCEPT_FIELD = '所属概念'
const COLLAPSED_CHIPS = 8

// 泛概念黑名单: 交易属性/指数成份类标签, 没有题材信息量
const GENERIC_EXACT = new Set([
  '融资融券', '转融券标的', '融资标的', '融券标的',
  '新股与次新股', '注册制次新股', '次新股',
  '标普道琼斯A股', '富时罗素概念', '富时罗素概念股', 'MSCI中国', 'MSCI概念',
  '深股通', '沪股通', '陆股通', 'B股概念', 'AH股', 'GDR', 'H股',
  '央视50', '破净股', '微盘股', '低价股', '高股息精选',
])
const GENERIC_SUFFIX = /(成份股|成分股|样本股|标的)$/

export function isGenericConcept(c: string): boolean {
  return GENERIC_EXACT.has(c) || GENERIC_SUFFIX.test(c)
}

/** 全市场 symbol → 概念列表。自选页与面板共用(筛选联动需要)。 */
export function useWatchlistConcepts(source: ConceptSource = 'kpl') {
  const tableId = CONCEPT_SOURCE_TABLE[source]
  const query = useQuery({
    queryKey: QK.extDataRows(tableId, undefined, CONCEPT_LIMIT),
    queryFn: () => api.extDataRows(tableId, { limit: CONCEPT_LIMIT }),
    staleTime: 30 * 60_000,
  })
  const symbolConcepts = useMemo(() => {
    const map = new Map<string, string[]>()
    for (const r of (query.data?.rows ?? []) as Record<string, unknown>[]) {
      const sym = r.symbol ?? r['股票代码']
      const s = r[CONCEPT_FIELD]
      if (!sym || typeof s !== 'string' || !s) continue
      map.set(String(sym), s.split(/[;；]/).map(x => x.trim()).filter(Boolean))
    }
    return map
  }, [query.data])
  return { query, symbolConcepts }
}

interface MemberStock {
  symbol: string
  name: string
  pct: number | null
}

interface ConceptGroup {
  concept: string
  members: MemberStock[]
  avgPct: number | null
}

interface MarketConceptStat {
  avgPct: number
  count: number
  rank: number
  total: number
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function avg(vals: number[]): number | null {
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
}

interface Props {
  rows: Record<string, any>[]
  symbolConcepts: Map<string, string[]>
  conceptsLoading: boolean
  selected: Set<string>
  onToggle: (concept: string) => void
  onClear: () => void
  onPreview?: (symbol: string, name: string) => void
  source: ConceptSource
  onSourceChange: (s: ConceptSource) => void
  /** 历史回看日期(YYYY-MM-DD)。为空=最新/实时。影响市场热度取哪天的全市场快照。 */
  asOf?: string | null
}

export function WatchlistConceptPanel({
  rows, symbolConcepts, conceptsLoading, selected, onToggle, onClear, onPreview,
  source, onSourceChange, asOf,
}: Props) {
  const qc = useQueryClient()
  const [open, setOpen] = useState<boolean>(() => storage.watchlistConceptPanel.get(false))
  const [showGeneric, setShowGeneric] = useState<boolean>(() => storage.watchlistConceptShowGeneric.get(false))
  const [sortMode, setSortMode] = useState<'count' | 'avgPct'>('count')

  const toggleOpen = () => setOpen(v => { storage.watchlistConceptPanel.set(!v); return !v })
  const toggleGeneric = () => setShowGeneric(v => { storage.watchlistConceptShowGeneric.set(!v); return !v })

  // 概念表为空(内置预设还没拉过数据) → 提供一键获取
  const conceptsEmpty = !conceptsLoading && symbolConcepts.size === 0
  const tableId = CONCEPT_SOURCE_TABLE[source]
  const fetchMutation = useMutation({
    mutationFn: () => api.extDataPresetFetch(tableId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QK.extData })
      qc.invalidateQueries({ queryKey: QK.extDataRows(tableId, undefined, CONCEPT_LIMIT) })
    },
  })

  // ── 自选内聚合: 概念 → 成员/平均涨幅 ──
  const groups = useMemo<ConceptGroup[]>(() => {
    const map = new Map<string, MemberStock[]>()
    for (const r of rows) {
      const sym = String(r.symbol ?? '')
      if (!sym) continue
      const concepts = symbolConcepts.get(sym)
      if (!concepts?.length) continue
      const member: MemberStock = {
        symbol: sym,
        name: String(r.rt_name ?? r.name ?? sym),
        pct: num(r.rt_pct) ?? num(r.change_pct),
      }
      for (const c of concepts) {
        if (!showGeneric && isGenericConcept(c)) continue
        const arr = map.get(c)
        if (arr) arr.push(member)
        else map.set(c, [member])
      }
    }
    const out: ConceptGroup[] = []
    for (const [concept, members] of map) {
      out.push({
        concept, members,
        avgPct: avg(members.map(m => m.pct).filter((v): v is number => v != null)),
      })
    }
    out.sort((a, b) =>
      sortMode === 'avgPct'
        ? (b.avgPct ?? -Infinity) - (a.avgPct ?? -Infinity)
        : b.members.length - a.members.length || (b.avgPct ?? -Infinity) - (a.avgPct ?? -Infinity),
    )
    return out
  }, [rows, symbolConcepts, showGeneric, sortMode])

  // ── 市场热度: 仅展开时请求全市场快照, 算每个概念全市场均涨与排名 ──
  const marketQuery = useQuery({
    queryKey: QK.marketSnapshot(asOf ?? undefined),
    queryFn: () => api.marketSnapshot(asOf ?? undefined),
    staleTime: 60_000,
    enabled: open && symbolConcepts.size > 0,
  })

  const marketStats = useMemo(() => {
    const snap = marketQuery.data?.rows
    if (!snap?.length) return null
    const pctBySymbol = new Map<string, number>()
    for (const r of snap as MarketSnapshotRow[]) {
      const p = num(r.change_pct)
      if (r.symbol && p != null) pctBySymbol.set(String(r.symbol), p)
    }
    const acc = new Map<string, { sum: number; n: number }>()
    for (const [sym, concepts] of symbolConcepts) {
      const p = pctBySymbol.get(sym)
      if (p == null) continue
      for (const c of concepts) {
        const a = acc.get(c)
        if (a) { a.sum += p; a.n += 1 }
        else acc.set(c, { sum: p, n: 1 })
      }
    }
    // 排名只在"非泛概念且成员数>=3"的集合里排, 避免被一两只票的迷你概念刷榜
    const ranked = [...acc.entries()]
      .filter(([c, a]) => a.n >= 3 && !isGenericConcept(c))
      .map(([c, a]) => ({ c, avgPct: a.sum / a.n }))
      .sort((x, y) => y.avgPct - x.avgPct)
    const out = new Map<string, MarketConceptStat>()
    ranked.forEach((r, i) => {
      const a = acc.get(r.c)!
      out.set(r.c, { avgPct: r.avgPct, count: a.n, rank: i + 1, total: ranked.length })
    })
    // 泛概念/小概念也给均值(无排名 rank=0)
    for (const [c, a] of acc) {
      if (!out.has(c)) out.set(c, { avgPct: a.sum / a.n, count: a.n, rank: 0, total: ranked.length })
    }
    return out
  }, [marketQuery.data, symbolConcepts])

  if (!rows.length) return null

  const chipClass = (active: boolean) =>
    `inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] leading-tight transition-colors cursor-pointer ${
      active
        ? 'bg-accent/20 text-accent ring-1 ring-accent/40'
        : 'bg-elevated text-secondary hover:text-foreground hover:bg-elevated/80'
    }`

  return (
    <div className="px-5 py-1.5 border-b border-border bg-surface/30">
      {/* 折叠条 */}
      <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={toggleOpen}
          className="inline-flex items-center gap-1 text-[11px] font-medium text-secondary hover:text-foreground transition-colors shrink-0"
        >
          {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          概念透视
          {groups.length > 0 && <span className="text-muted font-normal">({groups.length})</span>}
        </button>

        {/* 数据源切换: 开盘啦 / 同花顺 */}
        <div className="inline-flex items-center rounded bg-elevated p-0.5 shrink-0">
          {(['kpl', 'ths'] as const).map(s => (
            <button
              key={s}
              onClick={() => onSourceChange(s)}
              className={`px-1.5 py-0.5 rounded text-[10px] leading-tight transition-colors ${
                source === s ? 'bg-accent/20 text-accent font-medium' : 'text-muted hover:text-foreground'
              }`}
              title={s === 'kpl' ? '开盘啦题材库(时效/质量更好)' : '同花顺概念'}
            >
              {CONCEPT_SOURCE_LABEL[s]}
            </button>
          ))}
        </div>

        {conceptsLoading && <RefreshCw className="h-3 w-3 animate-spin text-muted" />}

        {conceptsEmpty && (
          <button
            onClick={() => fetchMutation.mutate()}
            disabled={fetchMutation.isPending}
            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] bg-accent/10 text-accent hover:bg-accent/20 transition-colors disabled:opacity-50"
          >
            {fetchMutation.isPending ? <RefreshCw className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
            概念数据未获取, 点此拉取
          </button>
        )}

        {/* 收起态: 前 N 个概念 chips */}
        {!open && groups.slice(0, COLLAPSED_CHIPS).map(g => (
          <button key={g.concept} onClick={() => onToggle(g.concept)} className={chipClass(selected.has(g.concept))}>
            <span>{g.concept}</span>
            <span className="font-mono text-[10px] opacity-70">×{g.members.length}</span>
            {g.avgPct != null && (
              <span className={`font-mono text-[10px] ${priceColorClass(g.avgPct)}`}>{fmtPct(g.avgPct, 1)}</span>
            )}
          </button>
        ))}
        {!open && groups.length > COLLAPSED_CHIPS && (
          <button onClick={toggleOpen} className="text-[10px] text-muted hover:text-accent transition-colors">
            +{groups.length - COLLAPSED_CHIPS} 更多…
          </button>
        )}

        {selected.size > 0 && (
          <button
            onClick={onClear}
            className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] bg-warning/12 text-warning border border-warning/25 hover:bg-warning/20 transition-colors"
          >
            <X className="h-2.5 w-2.5" /> 清除概念筛选 ({selected.size})
          </button>
        )}
      </div>

      {/* 展开态 */}
      {open && (
        <div className="mt-1.5">
          <div className="flex items-center gap-3 mb-1.5 text-[10px] text-muted">
            <span>排序:</span>
            {(['count', 'avgPct'] as const).map(m => (
              <button
                key={m}
                onClick={() => setSortMode(m)}
                className={`transition-colors ${sortMode === m ? 'text-accent font-medium' : 'hover:text-foreground'}`}
              >
                {m === 'count' ? '覆盖数' : '平均涨幅'}
              </button>
            ))}
            <span className="w-px h-3 bg-border" />
            <button onClick={toggleGeneric} className={`transition-colors ${showGeneric ? 'text-accent' : 'hover:text-foreground'}`}>
              {showGeneric ? '隐藏泛概念' : '显示泛概念'}
            </button>
            {open && marketQuery.isLoading && (
              <span className="inline-flex items-center gap-1"><RefreshCw className="h-2.5 w-2.5 animate-spin" />加载市场热度…</span>
            )}
          </div>

          <div className="max-h-72 overflow-y-auto pr-1 space-y-1">
            {groups.map(g => {
              const mk = marketStats?.get(g.concept)
              const active = selected.has(g.concept)
              return (
                <div key={g.concept} className={`rounded-md border px-2 py-1 transition-colors ${
                  active ? 'border-accent/40 bg-accent/5' : 'border-border/60 bg-surface/50'
                }`}>
                  <div className="flex items-center gap-2 flex-wrap">
                    <button onClick={() => onToggle(g.concept)} className={chipClass(active)}>
                      <span className="font-medium">{g.concept}</span>
                      <span className="font-mono text-[10px] opacity-70">×{g.members.length}</span>
                    </button>
                    <span className="text-[10px] text-muted">
                      自选均涨 <span className={`font-mono ${priceColorClass(g.avgPct)}`}>{fmtPct(g.avgPct, 2)}</span>
                    </span>
                    {mk && (
                      <span className="text-[10px] text-muted">
                        全市场({mk.count}只) <span className={`font-mono ${priceColorClass(mk.avgPct)}`}>{fmtPct(mk.avgPct, 2)}</span>
                        {mk.rank > 0 && (
                          <span className={`ml-1 font-mono ${mk.rank <= Math.max(10, mk.total * 0.1) ? 'text-warning' : ''}`}>
                            热度#{mk.rank}/{mk.total}
                          </span>
                        )}
                      </span>
                    )}
                  </div>
                  <div className="mt-0.5 flex flex-wrap gap-x-2 gap-y-0.5">
                    {g.members.map(m => (
                      <button
                        key={m.symbol}
                        onClick={() => onPreview?.(m.symbol, m.name)}
                        className="inline-flex items-center gap-1 text-[10px] text-secondary hover:text-accent transition-colors"
                        title={m.symbol}
                      >
                        <span>{m.name}</span>
                        <span className={`font-mono ${priceColorClass(m.pct)}`}>{fmtPct(m.pct, 1)}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )
            })}
            {groups.length === 0 && !conceptsEmpty && !conceptsLoading && (
              <div className="text-[11px] text-muted py-2">自选股没有命中任何概念{showGeneric ? '' : '(可尝试"显示泛概念")'}</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
