import raw from './pragma_demo.json'

// ============================================================
// ФАКТЫ ПРОЕКТА
// ============================================================
//
// Всё, что визуализация говорит о модели и данных, приходит из
// pragma_demo.json — его пишет viz/scripts/export_demo.py кодом
// проекта. Здесь только типы и удобный доступ; ни одного числа
// модели руками.
// ============================================================

export type TokenKind = 'special' | 'key' | 'value' | 'bucket' | 'bpe'

export interface BucketRange {
  key: string
  name: string
  min: number | null
  max: number | null
  when: string | null
}

export interface Token {
  key_id: number
  value_id: number
  position: number
  key: string
  value: string
  kind: TokenKind
  range?: BucketRange
  piece?: string
  text?: string
  time?: string | null
  time_log?: number
}

export interface TimelineEvent {
  t: string
  type: string
  source: string
  n_tokens: number
  time_log: number
  target: boolean
}

export interface EventView {
  index: number
  type: string
  source: string
  time: string
  local_time: string
  raw: Record<string, string | number | boolean | null>
  calendar: number[]
  time_log: number
  target: boolean
  tokens: Token[]
}

export interface EncoderInfo {
  config: Record<string, number | string>
  blocks: number
  parameters: number
  parts: Record<string, number>
}

export interface EpochSummary {
  epoch: number
  step: number
  train_loss: number | null
  val_loss: number | null
  learning_rate: number | null
  train_seconds: number | null
  val_seconds: number | null
  grad_norm_mean: number | null
  clipped_share: number | null
  cuda_peak_allocated_gib: number | null
}

export interface Interval {
  mean: number
  low: number
  high: number
}

export interface UsrDiagnosticTask {
  rows: number
  positives: number
  cells: Record<string, { pr_auc: number; roc_auc: number; f1: number }>
  deltas: Partial<Record<'usr_vs_control' | 'catboost_vs_lr', { pr_auc: Interval; roc_auc: Interval }>>
}

export interface Demo {
  format: number
  sources: { generated_at: string; vocabulary_digest: string; group: string }
  vocabulary: {
    size: number
    kinds: Record<TokenKind, { count: number; first: number; last: number }>
    specials: Record<string, number>
  }
  architecture: {
    dim: number
    vocab_size: number
    embedding_parameters: number
    head_parameters: number
    encoders: Record<'event' | 'profile' | 'history', EncoderInfo>
    total_parameters: number
  }
  training: Record<string, number | string | boolean | null>
  masking: Record<string, number | string | boolean | null>
  dataset: {
    group: string
    cutoff: string
    time_anchor: string
    window: Record<string, string>
    context: Record<string, number | string>
    profile_fields: string[]
    lifelong_types: string[]
    calendar: { cycles: Record<string, number>; features: string[] }
    bank_timezone: string
    bank_utc_offset_hours: number
  }
  batching: { clients: number; micro_batches: number; tokens: number; mean_tokens: number; max_tokens: number }
  client: {
    client_id: string
    n_events: number
    n_tokens: number
    n_profile_tokens: number
    dropped_old_events: number
    type_counts: Record<string, number>
    source_counts: Record<string, number>
    timeline: TimelineEvent[]
    highlighted: Record<'hero' | 'salary_credit' | 'app_screen' | 'communication_sent', EventView>
    mlm_candidates: { key: string; target: number; buckets: (BucketRange & { value_id: number })[] }
    profile: Token[]
  }
  downstream: {
    tasks: string[]
    reference: Record<string, string>
    compared: string[]
    readouts: string[]
    horizon_days: number
    cutoffs: Record<string, string>
    // CatBoost-бейзлайн задач churn на val текущей выгрузки.
    catboost_churn: {
      group: string
      tasks: Record<string, { pr_auc: number; roc_auc: number; rows: number; positives: number }>
      source: string
    } | null
    // Диагностика [USR] прогона на val: векторы модели и начальных
    // весов × регрессия и CatBoost только на [USR] (probe --control init).
    usr_diagnostic: {
      group: string
      tag: string
      control: string
      tasks: Record<string, UsrDiagnosticTask>
      source: string
    } | null
  }
  // Настоящие числа обученной модели (export_demo.py --checkpoint).
  model: {
    checkpoint: string
    epoch: number
    val_loss: number | null
    attention: { row_error: number; blocks: number[][][] }
    mlm: {
      event_index: number
      target: number
      target_probability: number
      target_rank: number
      top5: { value_id: number; name: string; p: number }[]
      candidates: Record<string, number>
    }
  } | null
  run: {
    run: string
    plan: Record<string, number | string | boolean | null> | null
    epochs: EpochSummary[]
    steps: number
    total_steps: number | null
    best_val_loss: number | null
    header: string[]
  } | null
}

export const demo = raw as unknown as Demo

export const hero = demo.client.highlighted.hero

export const DIM = demo.architecture.dim

// ------------------------------------------------------------
// Формат чисел: тонкий пробел в тысячах, точка в дробях.
// ------------------------------------------------------------

// Неразрывный пробел: он есть и в Inter, и в JetBrains Mono.
const THIN = '\u00a0'

export function int(value: number): string {
  return Math.round(value)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, THIN)
}

export function num(value: number | null | undefined, digits = 4): string {
  return value === null || value === undefined || !Number.isFinite(value) ? '—' : value.toFixed(digits)
}

export function sci(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : value.toExponential(0).replace('e-', 'e−')
}

export function range(bucket: BucketRange): string {
  const left = bucket.min === null ? '−∞' : int(bucket.min)
  const right = bucket.max === null ? '+∞' : int(bucket.max)
  return `[${left}, ${right})`
}

// Токены события без маркера, сгруппированные по ключу: значение
// из нескольких кусков BPE — одна строка.
export function pairs(event: EventView): { key: string; tokens: Token[] }[] {
  const out: { key: string; tokens: Token[] }[] = []

  for (const token of event.tokens) {
    if (token.kind === 'special') continue
    const last = out[out.length - 1]
    if (last && last.key === token.key && token.position > 0) last.tokens.push(token)
    else out.push({ key: token.key, tokens: [token] })
  }

  return out
}

// Короткое имя значения токена без префикса вида.
export function shortValue(token: Token): string {
  if (token.kind === 'special') return token.value
  // Кусок BPE — в кавычках: ведущий пробел в нём значим.
  if (token.kind === 'bpe') return `"${token.piece ?? ''}"`
  if (token.kind === 'value') return token.text ?? token.value
  return token.value.slice(token.value.indexOf(':') + 1)
}

// Дата в ISO → «2024-06-16».
export function day(iso: string): string {
  return iso.slice(0, 10)
}

// Местная дата банка: тот же фиксированный сдвиг, что у проекта.
const OFFSET_MS = demo.dataset.bank_utc_offset_hours * 3600 * 1000

export function localDay(iso: string): string {
  return new Date(Date.parse(iso) + OFFSET_MS).toISOString().slice(0, 10)
}

// Момент UTC для местной полуночи банка.
export function localMidnight(year: number, month: number, date = 1): number {
  return Date.UTC(year, month, date) - OFFSET_MS
}
