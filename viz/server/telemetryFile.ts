import { closeSync, openSync, readSync, statSync } from 'node:fs'

// ============================================================
// ЧТЕНИЕ telemetry.jsonl С МЕСТА ОСТАНОВКИ
// ============================================================
//
// Отдаются только целые строки: хвост без перевода строки
// обучение ещё пишет. Файл короче прочитанного или другой файл
// на том же месте (identity = dev:ino) — новое обучение стёрло
// прежний, и чтение начинается с нуля (reset). За раз — не
// больше limit байт: клиент дочитывает циклом, пока next < size.
// ============================================================

export const CHUNK = 4 << 20

export interface Chunk {
  missing?: true
  identity: string
  size: number
  mtimeMs: number
  reset: boolean
  offset: number
  next: number
  text: string
}

export function readComplete(path: string, offset: number, identity: string | null, limit = CHUNK): Chunk {
  let stats
  try {
    stats = statSync(path)
  } catch {
    return { missing: true, identity: '', size: 0, mtimeMs: 0, reset: offset > 0, offset: 0, next: 0, text: '' }
  }

  const current = `${stats.dev}:${stats.ino}`
  const reset = (identity !== null && identity !== current) || stats.size < offset
  const start = reset ? 0 : Math.max(0, offset)

  if (stats.size <= start) {
    return { identity: current, size: stats.size, mtimeMs: stats.mtimeMs, reset, offset: start, next: start, text: '' }
  }

  const length = Math.min(limit, stats.size - start)
  const buffer = Buffer.alloc(length)
  const handle = openSync(path, 'r')

  try {
    readSync(handle, buffer, 0, length, start)
  } finally {
    closeSync(handle)
  }

  // Только до последнего перевода строки.
  const end = buffer.lastIndexOf(0x0a)

  if (end < 0) {
    return { identity: current, size: stats.size, mtimeMs: stats.mtimeMs, reset, offset: start, next: start, text: '' }
  }

  return {
    identity: current,
    size: stats.size,
    mtimeMs: stats.mtimeMs,
    reset,
    offset: start,
    next: start + end + 1,
    text: buffer.subarray(0, end + 1).toString('utf8'),
  }
}
