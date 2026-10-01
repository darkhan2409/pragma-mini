import { describe, expect, test } from 'vitest'
import { demo, hero, int, localDay, pairs } from './demo'

// ============================================================
// Экспорт фактов (pragma_demo.json): сверки, на которых держится
// честность визуализации. Сам экспорт проверяется ещё и тестом
// tests/test_viz_export.py на синтетическом мире.
// ============================================================

describe('словарь и модель', () => {
  test('виды токенов покрывают словарь, номера идут подряд', () => {
    const kinds = Object.values(demo.vocabulary.kinds)
    expect(kinds.reduce((sum, kind) => sum + kind.count, 0)).toBe(demo.vocabulary.size)
    const ordered = [...kinds].sort((a, b) => a.first - b.first)
    ordered.forEach((kind, index) => {
      expect(kind.last - kind.first + 1).toBe(kind.count)
      if (index > 0) expect(kind.first).toBe(ordered[index - 1].last + 1)
    })
  })

  test('параметры: таблица + голова + энкодеры = всего', () => {
    const arch = demo.architecture
    expect(arch.embedding_parameters).toBe(arch.vocab_size * arch.dim)
    expect(arch.head_parameters).toBe(3 * arch.dim * arch.dim + arch.dim)
    const encoders = Object.values(arch.encoders).reduce((sum, item) => sum + item.parameters, 0)
    expect(arch.total_parameters).toBe(arch.embedding_parameters + arch.head_parameters + encoders)
    for (const item of Object.values(arch.encoders)) {
      expect(Object.values(item.parts).reduce((a, b) => a + b, 0)).toBe(item.parameters)
    }
  })

  test('head_dim = d / heads у истории', () => {
    const history = demo.architecture.encoders.history
    expect(demo.architecture.dim % Number(history.config.heads)).toBe(0)
  })
})

describe('клиент', () => {
  const client = demo.client

  test('лента совпадает со счётами экспорта', () => {
    expect(client.timeline).toHaveLength(client.n_events)
    expect(client.timeline.reduce((sum, item) => sum + item.n_tokens, 0)).toBe(client.n_tokens)
    expect(client.profile).toHaveLength(client.n_profile_tokens)
    expect(Object.values(client.type_counts).reduce((a, b) => a + b, 0)).toBe(client.n_events)
  })

  test('лента по времени, давность до T не растёт к концу', () => {
    for (let i = 1; i < client.timeline.length; i++) {
      expect(Date.parse(client.timeline[i].t)).toBeGreaterThanOrEqual(Date.parse(client.timeline[i - 1].t))
      expect(client.timeline[i].time_log).toBeLessThanOrEqual(client.timeline[i - 1].time_log + 1e-6)
    }
  })

  test('сквозное событие — покупка: [EVT] первым, название из нескольких кусков BPE', () => {
    expect(hero.type).toBe('purchase')
    expect(hero.tokens[0].value).toBe('[EVT]')
    const name = pairs(hero).find((row) => row.key === 'merchant_name')!
    expect(name.tokens.length).toBeGreaterThanOrEqual(2)
    expect(name.tokens.map((token) => token.position)).toEqual(name.tokens.map((_, index) => index))
    expect(name.tokens.every((token) => token.kind === 'bpe')).toBe(true)
    expect(hero.calendar).toHaveLength(6)
  })

  test('сумма попала в свой диапазон', () => {
    const amount = Number(hero.raw.transaction_amount)
    const token = hero.tokens.find((item) => item.key === 'transaction_amount')!
    expect(token.kind).toBe('bucket')
    const range = token.range!
    if (range.min !== null) expect(amount).toBeGreaterThanOrEqual(range.min)
    if (range.max !== null) expect(amount).toBeLessThan(range.max)
    expect(range.when).toBe(hero.raw.direction)
  })

  test('цель MLM — среди кандидатов той же шкалы', () => {
    const candidates = client.mlm_candidates
    expect(candidates.buckets.some((bucket) => bucket.value_id === candidates.target)).toBe(true)
    expect(candidates.buckets.every((bucket) => bucket.key === candidates.key)).toBe(true)
  })

  test('анкета: [USR] первым с давностью 0', () => {
    expect(client.profile[0].value).toBe('[USR]')
    expect(client.profile[0].time_log).toBe(0)
  })
})

describe('формат', () => {
  test('тысячи через неразрывный пробел', () => {
    expect(int(18510)).toBe('18 510')
    expect(int(617)).toBe('617')
  })

  test('местная дата банка — тем же сдвигом, что у проекта', () => {
    expect(localDay(demo.dataset.cutoff)).toBe('2026-01-01')
  })
})
