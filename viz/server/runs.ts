import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { addLines, emptyTelemetry, progress } from '../src/telemetry/model.ts'

// ============================================================
// ПРОГОНЫ И СТАТУСЫ ВОЛНЫ 4
// ============================================================
//
// Прогоны — data/12_train и каталоги data/runs/*. Статус
// эксперимента выводится только из файлов прогона:
//
//   NOT_RUN  нет каталога или телеметрии;
//   RUNNING  эпохи не все, файл менялся меньше 10 минут назад;
//   DONE     все запланированные эпохи или строка остановки в
//            train.log.
//
// ACCEPTED и REJECTED сервер не ставит: это решение человека
// (src/data/wave4_decisions.json).
// ============================================================

export const TELEMETRY = 'telemetry.jsonl'

export function dataDir(): string {
  return process.env.PRAGMA_DATA_DIR ?? resolve(process.cwd(), '..', 'data')
}

export interface RunInfo {
  id: string
  path: string
  telemetry: boolean
  size: number
  mtimeMs: number
}

export function discoverRuns(root = dataDir()): RunInfo[] {
  const found: RunInfo[] = []

  const candidates: [string, string][] = [['12_train', join(root, '12_train')]]

  const runs = join(root, 'runs')
  if (existsSync(runs)) {
    for (const name of readdirSync(runs).sort()) {
      const path = join(runs, name)
      if (statSync(path).isDirectory()) candidates.push([`runs/${name}`, path])
    }
  }

  for (const [id, path] of candidates) {
    if (!existsSync(path)) continue
    const file = join(path, TELEMETRY)
    const stats = existsSync(file) ? statSync(file) : null
    found.push({ id, path, telemetry: stats !== null, size: stats?.size ?? 0, mtimeMs: stats?.mtimeMs ?? 0 })
  }

  return found.sort((a, b) => Number(b.telemetry) - Number(a.telemetry) || b.mtimeMs - a.mtimeMs)
}

export type Status = 'NOT_RUN' | 'RUNNING' | 'DONE' | 'STALLED'

export interface RunStatus {
  id: string
  status: Status
  epoch: number | null
  epochs: number | null
  step: number
  totalSteps: number | null
  bestValLoss: number | null
  minutesSinceWrite: number | null
  checkpoint: boolean
  stopReason: string | null
}

const STALE_MS = 10 * 60 * 1000

// Статус одного каталога прогона. now — для тестов.
export function runStatus(id: string, path: string, now = Date.now()): RunStatus {
  const empty: RunStatus = {
    id,
    status: 'NOT_RUN',
    epoch: null,
    epochs: null,
    step: 0,
    totalSteps: null,
    bestValLoss: null,
    minutesSinceWrite: null,
    checkpoint: false,
    stopReason: null,
  }

  const file = join(path, TELEMETRY)
  if (!existsSync(file)) return empty

  const telemetry = emptyTelemetry()
  addLines(telemetry, readFileSync(file, 'utf8'))
  const where = progress(telemetry)

  const log = join(path, 'train.log')
  let stopReason: string | null = null
  if (existsSync(log)) {
    const match = /остановка: ([a-z_]+)/.exec(readFileSync(log, 'utf8').split('\n').slice(-40).join('\n'))
    stopReason = match ? match[1] : null
  }

  const age = now - statSync(file).mtimeMs
  const complete = where.epochs !== null && where.completedEpochs >= where.epochs

  let status: Status = 'STALLED'
  if (complete || stopReason) status = 'DONE'
  else if (age < STALE_MS) status = 'RUNNING'

  return {
    id,
    status,
    epoch: where.epoch,
    epochs: where.epochs,
    step: where.step,
    totalSteps: where.totalSteps,
    bestValLoss: where.bestValLoss,
    minutesSinceWrite: Math.round(age / 60000),
    checkpoint: existsSync(join(path, 'best_checkpoint.pt')),
    stopReason,
  }
}

// Эксперименты волны 4: каталоги data/runs/w4-<id>.
export function wave4Statuses(ids: string[], root = dataDir()): RunStatus[] {
  return ids.map((id) => {
    const path = join(root, 'runs', `w4-${id.toLowerCase()}`)
    return existsSync(path) ? runStatus(id, path) : { ...runStatus(id, join(path, 'missing')), id }
  })
}
