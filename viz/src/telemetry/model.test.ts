import { describe, expect, test } from 'vitest'
import { add, addLines, emptyTelemetry, kpis, progress, pyExp, roundHalfEven, thin } from './model'

// ============================================================
// Порт читателя телеметрии: те же случаи, что tests/test_dashboard.py
// (Python-читатель и Streamlit-страница), и те же строки KPI.
// ============================================================

function run(epoch = 1, step = 0, epochs = 2, total = 4, resumed = false, planned?: number) {
  return { kind: 'run', epoch, step, epochs, planned_epochs: planned ?? epochs, total_steps: total, max_steps: null, max_grad_norm: 1.0, resumed, time: 0 }
}

function step(number: number, epoch = 1, loss = 2.0) {
  return { kind: 'step', epoch, step: number, loss, targets: 10, micro_batches: 1, tokens: 100, lr: 1e-4, grad_norm: 0.5, wait_seconds: 0, seconds: 0.5, time: 0 }
}

function epoch(number: number, at: number, val: number | null = 1.5) {
  return { kind: 'epoch', epoch: number, step: at, train_loss: 2.0, val_loss: val }
}

function lines(...records: object[]): string {
  return records.map((record) => JSON.stringify(record) + '\n').join('')
}

describe('разбор строк', () => {
  test('битая строка пропускается и считается, соседние целы', () => {
    const telemetry = emptyTelemetry()
    addLines(telemetry, lines(run(), step(1)) + JSON.stringify(step(2)).slice(0, 20) + lines(run(1, 1, 2, 4, true), step(2, 1, 1.0)))
    expect(telemetry.skipped).toBe(1)
    expect([...telemetry.steps.keys()].sort()).toEqual([1, 2])
    expect(telemetry.steps.get(2)!.loss).toBe(1.0)
  })

  test('продолжение отбрасывает записанное после последнего чекпойнта', () => {
    const telemetry = emptyTelemetry()
    addLines(telemetry, lines(run(), step(1), step(2), step(3), epoch(1, 3), step(4, 2), step(5, 2), epoch(2, 5, 9.9), run(2, 3, 2, 4, true)))
    expect([...telemetry.steps.keys()].sort()).toEqual([1, 2, 3])
    expect([...telemetry.epochs.keys()]).toEqual([1])

    addLines(telemetry, lines(step(4, 2, 1.25), epoch(2, 4, 1.1)))
    const where = progress(telemetry)
    expect(telemetry.steps.get(4)!.loss).toBe(1.25)
    expect(where.valLoss).toBe(1.1)
    expect(where.bestValLoss).toBe(1.1)
    expect(where.resumes).toBe(1)
    expect(where.completedEpochs).toBe(2)
  })

  test('не объект и неизвестный вид — пропуск без изменений', () => {
    const telemetry = emptyTelemetry()
    add(telemetry, [1, 2])
    add(telemetry, { kind: 'mystery', step: 1 })
    add(telemetry, { kind: 'step', step: 'x' })
    expect(telemetry.skipped).toBe(3)
    expect(telemetry.steps.size).toBe(0)
  })
})

describe('прогресс', () => {
  test('эпохи сверх плана — оценка, max_steps ограничивает', () => {
    const telemetry = emptyTelemetry()
    add(telemetry, run(1, 0, 3, 4, false, 2))
    expect(progress(telemetry).totalSteps).toBe(6)
    expect(progress(telemetry).estimated).toBe(true)

    add(telemetry, { ...run(1, 0, 2, 4), max_steps: 3 })
    expect(progress(telemetry).totalSteps).toBe(3)
  })

  test('округление к чётному, как round() в Python', () => {
    expect(roundHalfEven(2.5)).toBe(2)
    expect(roundHalfEven(3.5)).toBe(4)
    expect(roundHalfEven(3.4)).toBe(3)
  })
})

describe('KPI как на Streamlit-странице', () => {
  test('строки плиток совпадают со страницей tests/test_dashboard.py', () => {
    const telemetry = emptyTelemetry()
    addLines(
      telemetry,
      lines(
        run(1, 0, 2, 4),
        ...[1, 2, 3, 4].map((n) => step(n, 1 + Number(n > 2), 3.0 - n / 4)),
        epoch(1, 2, 2.5),
        { ...epoch(2, 4, 2.25), cuda_peak_allocated_gib: 1.5, cuda_peak_reserved_gib: 2.0 },
      ),
    )
    const k = kpis(telemetry)
    expect(k.epoch).toBe('2 / 2')
    expect(k.step).toBe('4 / 4')
    expect(k.trainLoss).toBe('2.0000')
    expect(k.valLoss).toBe('2.2500')
    expect(k.vram).toBe('1.50 ГиБ')
    expect(k.lr).toBe('1.00e-04')
    expect(k.tokensPerSecond).toBe('200')
  })

  test('порядок числа — две цифры, как format(x, ".2e")', () => {
    expect(pyExp(0.0003)).toBe('3.00e-04')
    expect(pyExp(12345)).toBe('1.23e+04')
  })
})

describe('свёртка точек', () => {
  test('не больше предела, шаг — последний в окне, значения — среднее', () => {
    const rows = Array.from({ length: 5000 }, (_, i) => ({ step: i + 1, loss: i % 2 }))
    const out = thin(rows, 2000, ['loss'])
    expect(out.length).toBeLessThanOrEqual(2000)
    expect(out[0].step).toBe(3)
    expect(out[0].loss).toBeCloseTo(1 / 3)
    expect(out[out.length - 1].step).toBe(5000)
  })

  test('короткий ряд не трогается', () => {
    const rows = [{ step: 1, loss: 1 }]
    expect(thin(rows, 2000, ['loss'])).toBe(rows)
  })
})
