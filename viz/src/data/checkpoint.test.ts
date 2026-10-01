import { describe, expect, test } from 'vitest'
import { attention, candidates, SHOWN } from './checkpoint'
import { demo, hero } from './demo'

// ============================================================
// Шаги 10 и 11 показывают числа чекпойнта без искажений: веса —
// строки внимания из экспорта, голова и события — по заявленному
// правилу, строки предсказания — top-5 словаря и цель.
// ============================================================

const model = demo.model
const n = demo.client.n_events

describe('внимание [USR]', () => {
  const view = attention()
  const indices = view.events.map((event) => event.index)

  test('разные события по времени, сквозное среди них, выборка по всей истории', () => {
    expect(indices).toHaveLength(Math.min(SHOWN, n))
    expect(new Set(indices).size).toBe(indices.length)
    expect([...indices].sort((a, b) => a - b)).toEqual(indices)
    expect(indices).toContain(hero.index)
    expect(indices[0]).toBeLessThan(n / SHOWN)
    expect(indices[indices.length - 1]).toBeGreaterThan(n - 1 - n / SHOWN)
    expect(Math.max(...view.events.map((event) => event.weight))).toBe(1)
  })

  test.runIf(model !== null)('веса — строка головы последнего блока с самым большим весом события', () => {
    const blocks = model!.attention.blocks
    expect(view.real).toBe(true)
    expect(view.block).toBe(blocks.length - 1)

    const rows = blocks[view.block]
    for (const row of rows) {
      expect(row).toHaveLength(n + 1)
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 4)
    }

    const peak = (row: number[]) => Math.max(...row.slice(1))
    expect(peak(rows[view.head])).toBe(Math.max(...rows.map(peak)))
    expect(view.max).toBe(peak(rows[view.head]))
    for (const event of view.events) expect(event.weight * view.max).toBeCloseTo(rows[view.head][1 + event.index], 12)

    const heaviest = rows[view.head]
      .slice(1)
      .map((weight, index) => [weight, index])
      .sort((a, b) => b[0] - a[0])
      .slice(0, SHOWN / 2)
      .map(([, index]) => index)
    expect(view.heaviest).toEqual(heaviest)
    for (const index of heaviest) expect(indices).toContain(index)
  })
})

describe('предсказание MLM', () => {
  const rows = candidates()

  test('цель одна: корзина, в которую попадает сумма сквозного события', () => {
    const targets = rows.filter((row) => row.target)
    expect(targets).toHaveLength(1)
    expect(targets[0].value_id).toBe(demo.client.mlm_candidates.target)

    const amount = hero.raw.transaction_amount as number
    const bucket = targets[0].bucket!
    expect(bucket.min === null || bucket.min <= amount).toBe(true)
    expect(bucket.max === null || amount <= bucket.max).toBe(true)
  })

  test.runIf(model !== null)('top-5 словаря по убыванию, затем цель на своём месте', () => {
    const mlm = model!.mlm
    expect(mlm.event_index).toBe(hero.index)
    expect(mlm.target).toBe(demo.client.mlm_candidates.target)

    expect(rows.slice(0, 5).map((row) => [row.value_id, row.p])).toEqual(mlm.top5.map((item) => [item.value_id, item.p]))
    for (let i = 1; i < rows.length; i++) expect(rows[i].p).toBeLessThanOrEqual(rows[i - 1].p)
    expect(rows.filter((row) => row.top).map((row) => row.value_id)).toEqual([mlm.top5[0].value_id])
    expect(rows).toHaveLength(mlm.target_rank <= 5 ? 5 : 6)

    // Две записи экспорта согласованы между собой.
    expect(rows.find((row) => row.target)!.p).toBe(mlm.target_probability)
    expect(mlm.candidates[mlm.target]).toBe(mlm.target_probability)
    const above = Object.values(mlm.candidates).filter((p) => p > mlm.target_probability).length
    expect(mlm.target_rank).toBeGreaterThanOrEqual(above + 1)
    expect(mlm.top5[0].p).toBeGreaterThanOrEqual(Math.max(...Object.values(mlm.candidates)))

    // Корзина строки — тот же токен, что назван в экспорте.
    for (const row of rows) if (row.bucket) expect(row.name.endsWith(row.bucket.name)).toBe(true)
  })
})
