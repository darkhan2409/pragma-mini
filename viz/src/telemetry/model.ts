// ============================================================
// ТЕЛЕМЕТРИЯ ОБУЧЕНИЯ: ПОРТ src/dashboard/telemetry.py
// ============================================================
//
// Та же семантика, что у Python-читателя и Streamlit-дашборда:
//
//   run    начало прогона и продолжения: план и горизонт шагов;
//   step   шаг оптимизатора: loss, lr, норма до клипа, токены;
//   epoch  полная эпоха: train/val loss, время, пики CUDA.
//
// Строка run продолжения отбрасывает шаги после своего шага и
// эпохи с её эпохи: их писал оборвавшийся прогон. Битая строка
// пропускается и считается. Тесты — model.test.ts, по образцу
// tests/test_dashboard.py.
// ============================================================

export type Record_ = Record<string, unknown>

export interface RunRecord extends Record_ {
  kind: 'run'
  epoch: number
  step: number
  epochs: number
  planned_epochs: number
  total_steps: number
  max_steps: number | null
  max_grad_norm?: number
  resumed?: boolean
}

export interface StepRecord extends Record_ {
  epoch: number
  step: number
  loss: number
  lr: number
  grad_norm: number
  tokens: number
  seconds: number
  targets?: number
  wait_seconds?: number
}

export interface EpochRecord extends Record_ {
  epoch: number
  step: number
  train_loss?: number | null
  val_loss?: number | null
  learning_rate?: number | null
  cuda_peak_allocated_gib?: number | null
  cuda_peak_reserved_gib?: number | null
  clipped_share?: number | null
  grad_norm_mean?: number | null
  grad_norm_max?: number | null
  train_seconds?: number
  val_seconds?: number
}

export interface Telemetry {
  runs: RunRecord[]
  steps: Map<number, StepRecord>
  epochs: Map<number, EpochRecord>
  skipped: number
  modified: number | null
}

export function emptyTelemetry(): Telemetry {
  return { runs: [], steps: new Map(), epochs: new Map(), skipped: 0, modified: null }
}

function integer(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) return Math.trunc(value)
  if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) return Math.trunc(Number(value))
  throw new TypeError(`не целое: ${String(value)}`)
}

// Одна запись. Ошибка разбора — пропуск без частичных изменений.
export function add(telemetry: Telemetry, record: unknown): void {
  if (record === null || typeof record !== 'object' || Array.isArray(record)) {
    telemetry.skipped += 1
    return
  }

  const item = record as Record_
  const kind = item.kind

  try {
    if (kind === 'run') {
      const step = integer(item.step)
      const epoch = integer(item.epoch)
      for (const number of [...telemetry.steps.keys()]) if (number > step) telemetry.steps.delete(number)
      for (const number of [...telemetry.epochs.keys()]) if (number >= epoch) telemetry.epochs.delete(number)
      telemetry.runs.push(item as RunRecord)
    } else if (kind === 'step') {
      telemetry.steps.set(integer(item.step), item as StepRecord)
    } else if (kind === 'epoch') {
      telemetry.epochs.set(integer(item.epoch), item as EpochRecord)
    } else {
      telemetry.skipped += 1
    }
  } catch {
    telemetry.skipped += 1
  }
}

// Целые строки текста; пустые — пропуск, битые — счёт.
export function addLines(telemetry: Telemetry, text: string): void {
  for (const line of text.split('\n')) {
    if (!line.trim()) continue
    let record: unknown
    try {
      record = JSON.parse(line)
    } catch {
      telemetry.skipped += 1
      continue
    }
    add(telemetry, record)
  }
}

// Округление к чётному, как round() в Python.
export function roundHalfEven(value: number): number {
  const floor = Math.floor(value)
  const diff = value - floor
  if (diff > 0.5) return floor + 1
  if (diff < 0.5) return floor
  return floor % 2 === 0 ? floor : floor + 1
}

export interface Progress {
  epoch: number | null
  epochs: number | null
  step: number
  totalSteps: number | null
  estimated: boolean
  completedEpochs: number
  lastStep: StepRecord | null
  lastEpoch: EpochRecord | null
  valLoss: number | null
  bestValLoss: number | null
  resumes: number
}

function last<T>(map: Map<number, T>): T | null {
  if (map.size === 0) return null
  return map.get(Math.max(...map.keys())) ?? null
}

export function progress(telemetry: Telemetry): Progress {
  const run = telemetry.runs.length ? telemetry.runs[telemetry.runs.length - 1] : null
  const lastStep = last(telemetry.steps)
  const lastEpoch = last(telemetry.epochs)

  const step = Math.max(0, ...[run, lastStep, lastEpoch].filter((item) => item !== null).map((item) => Number(item!.step)))

  let epoch: number | null = null
  if (lastStep) epoch = Number(lastStep.epoch)
  else if (run) epoch = Number(run.epoch)

  let totalSteps: number | null = null
  let estimated = false

  if (run) {
    totalSteps = Number(run.total_steps)
    if (Number(run.epochs) > Number(run.planned_epochs) && Number(run.planned_epochs)) {
      totalSteps = roundHalfEven((totalSteps * Number(run.epochs)) / Number(run.planned_epochs))
      estimated = true
    }
    if (run.max_steps !== null && run.max_steps !== undefined) totalSteps = Math.min(totalSteps, Number(run.max_steps))
  }

  const val = [...telemetry.epochs.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, record]) => record.val_loss)
    .filter((value): value is number => value !== null && value !== undefined)

  return {
    epoch,
    epochs: run ? Number(run.epochs) : null,
    step,
    totalSteps,
    estimated,
    completedEpochs: telemetry.epochs.size,
    lastStep,
    lastEpoch,
    valLoss: val.length ? val[val.length - 1] : null,
    bestValLoss: val.length ? Math.min(...val) : null,
    resumes: telemetry.runs.filter((item) => item.resumed).length,
  }
}

// ------------------------------------------------------------
// KPI как на Streamlit-странице (src/dashboard/app.py)
// ------------------------------------------------------------

// Как format(value, ".2e") в Python: порядок не короче двух цифр.
export function pyExp(value: number, digits = 2): string {
  const [mantissa, exponent] = value.toExponential(digits).split('e')
  const sign = exponent.startsWith('-') ? '-' : '+'
  return `${mantissa}e${sign}${exponent.replace(/^[+-]/, '').padStart(2, '0')}`
}

function grouped(value: number): string {
  return Math.round(value)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ',')
}

function fixed(value: number | null | undefined, digits: number): string {
  return value === null || value === undefined ? '—' : value.toFixed(digits)
}

export interface Kpis {
  epoch: string
  step: string
  trainLoss: string
  valLoss: string
  bestValLoss: string
  lr: string
  gradNorm: string
  tokensPerSecond: string
  vram: string
  vramReserved: string
}

export function median(values: number[]): number | null {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)]
}

export function kpis(telemetry: Telemetry): Kpis {
  const where = progress(telemetry)
  const lastStep = where.lastStep
  const recent = [...telemetry.steps.keys()].sort((a, b) => a - b).slice(-20).map((key) => telemetry.steps.get(key)!)
  const speed = median(recent.filter((item) => item.seconds && item.tokens).map((item) => item.tokens / item.seconds))
  const total = where.totalSteps === null ? '—' : `${where.estimated ? '≈' : ''}${grouped(where.totalSteps)}`
  const memory = where.lastEpoch ?? {}

  return {
    epoch: `${where.epoch ?? '—'} / ${where.epochs ?? '—'}`,
    step: `${grouped(where.step)} / ${total}`,
    trainLoss: fixed(lastStep?.loss, 4),
    valLoss: fixed(where.valLoss, 4),
    bestValLoss: fixed(where.bestValLoss, 4),
    lr: lastStep ? pyExp(lastStep.lr) : '—',
    gradNorm: fixed(lastStep?.grad_norm, 3),
    tokensPerSecond: speed === null ? '—' : grouped(speed),
    vram: `${fixed((memory as EpochRecord).cuda_peak_allocated_gib, 2)} ГиБ`,
    vramReserved: `${fixed((memory as EpochRecord).cuda_peak_reserved_gib, 2)} ГиБ`,
  }
}

// Не больше limit точек: соседние шаги — средним, шаг — последний в окне.
export function thin<T extends { step: number }>(rows: T[], limit = 2000, keys: (keyof T)[] = []): T[] {
  if (rows.length <= limit) return rows
  const size = Math.ceil(rows.length / limit)
  const out: T[] = []
  for (let i = 0; i < rows.length; i += size) {
    const chunk = rows.slice(i, i + size)
    const row = { ...chunk[chunk.length - 1] }
    for (const key of keys) {
      const values = chunk.map((item) => item[key] as unknown as number).filter((value) => Number.isFinite(value))
      ;(row as Record<string, unknown>)[key as string] = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null
    }
    out.push(row)
  }
  return out
}
