import { appendFileSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, test } from 'vitest'
import { readComplete } from './telemetryFile'
import { runStatus } from './runs'

// ============================================================
// Сервер /api: целые строки с места остановки, reset при новом
// файле, статусы прогонов. Только временные каталоги, не data/.
// ============================================================

let dir: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'pragma-viz-'))
})

afterEach(() => {
  rmSync(dir, { recursive: true, force: true })
})

describe('readComplete', () => {
  test('недописанная строка не отдаётся, пока не допишется', () => {
    const path = join(dir, 'telemetry.jsonl')
    writeFileSync(path, '{"a":1}\n{"b":')

    const first = readComplete(path, 0, null)
    expect(first.text).toBe('{"a":1}\n')
    expect(first.next).toBe(8)

    const again = readComplete(path, first.next, first.identity)
    expect(again.text).toBe('')
    expect(again.next).toBe(8)

    appendFileSync(path, '2}\n')
    const done = readComplete(path, again.next, first.identity)
    expect(done.text).toBe('{"b":2}\n')
    expect(done.reset).toBe(false)
  })

  test('файл короче прочитанного — чтение с нуля', () => {
    const path = join(dir, 'telemetry.jsonl')
    writeFileSync(path, '{"a":1}\n{"a":2}\n')
    const first = readComplete(path, 0, null)
    writeFileSync(path, '{"z":1}\n')
    const next = readComplete(path, first.next, first.identity)
    expect(next.reset).toBe(true)
    expect(next.text).toBe('{"z":1}\n')
  })

  test('другой файл на том же месте — чтение с нуля', () => {
    const path = join(dir, 'telemetry.jsonl')
    writeFileSync(path, '{"a":1}\n')
    const first = readComplete(path, 0, null)
    unlinkSync(path)
    writeFileSync(path, '{"a":1}\n{"b":2}\n')
    const next = readComplete(path, first.next, 'другой:0')
    expect(next.reset).toBe(true)
    expect(next.offset).toBe(0)
  })

  test('за раз не больше предела, остаток — следующим запросом', () => {
    const path = join(dir, 'telemetry.jsonl')
    writeFileSync(path, '{"i":1}\n{"i":2}\n{"i":3}\n')
    const first = readComplete(path, 0, null, 12)
    expect(first.text).toBe('{"i":1}\n')
    const second = readComplete(path, first.next, first.identity, 100)
    expect(second.text).toBe('{"i":2}\n{"i":3}\n')
    expect(second.next).toBe(second.size)
  })

  test('нет файла — missing', () => {
    expect(readComplete(join(dir, 'нет.jsonl'), 0, null).missing).toBe(true)
  })
})

describe('статус прогона', () => {
  const plan = (epochs: number) => JSON.stringify({ kind: 'run', epoch: 1, step: 0, epochs, planned_epochs: epochs, total_steps: 4, max_steps: null })

  test('нет телеметрии — NOT_RUN', () => {
    expect(runStatus('E1', dir).status).toBe('NOT_RUN')
  })

  test('свежие записи без всех эпох — RUNNING, давние — STALLED', () => {
    writeFileSync(join(dir, 'telemetry.jsonl'), plan(2) + '\n' + JSON.stringify({ kind: 'epoch', epoch: 1, step: 2, val_loss: 2 }) + '\n')
    expect(runStatus('B0', dir).status).toBe('RUNNING')
    expect(runStatus('B0', dir, Date.now() + 60 * 60 * 1000).status).toBe('STALLED')
  })

  test('все эпохи — DONE с лучшим val loss', () => {
    writeFileSync(
      join(dir, 'telemetry.jsonl'),
      [plan(2), JSON.stringify({ kind: 'epoch', epoch: 1, step: 2, val_loss: 2.1 }), JSON.stringify({ kind: 'epoch', epoch: 2, step: 4, val_loss: 1.9 })].join('\n') + '\n',
    )
    const status = runStatus('B0', dir, Date.now() + 60 * 60 * 1000)
    expect(status.status).toBe('DONE')
    expect(status.bestValLoss).toBe(1.9)
    // Строк шагов нет: эпоха — из строки run, как в progress() Python.
    expect(status.epoch).toBe(1)
    expect(status.epochs).toBe(2)
  })

  test('строка остановки в train.log — DONE', () => {
    writeFileSync(join(dir, 'telemetry.jsonl'), plan(5) + '\n')
    writeFileSync(join(dir, 'train.log'), '[train] эпох 3, шагов 12 на cuda, остановка: early_stopping, best_val_loss 1.9\n')
    const status = runStatus('B0', dir, Date.now() + 60 * 60 * 1000)
    expect(status.status).toBe('DONE')
    expect(status.stopReason).toBe('early_stopping')
  })
})
