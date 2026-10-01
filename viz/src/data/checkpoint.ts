import { demo, hero, type BucketRange } from './demo'

// ============================================================
// ЧИСЛА ШАГОВ 10 И 11: ВНИМАНИЕ [USR] И ПРЕДСКАЗАНИЕ MLM
// ============================================================
//
// С экспортом --checkpoint — настоящие числа обученной модели:
// веса внимания [USR] последнего блока истории и вероятности
// токенов для скрытой суммы сквозного события. Без чекпойнта —
// схема под плашкой «ИЛЛЮСТРАЦИЯ».
// ============================================================

// Сколько событий из истории показывает шаг 10.
export const SHOWN = 24

export interface Attention {
  // По времени; вес — доля от самого большого среди показанных.
  events: { index: number; weight: number }[]
  // С чекпойнтом — самые весомые события головы по убыванию веса.
  heaviest: number[]
  real: boolean
  block: number
  head: number
  max: number
}

// С чекпойнтом: голова последнего блока с самым большим весом
// одного события, её 12 самых весомых событий и равномерная
// выборка по всей истории. Без него — равномерная выборка и
// схема: ближе к T и у зарплаты вес больше.
export function attention(): Attention {
  const timeline = demo.client.timeline
  const blocks = demo.model?.attention.blocks

  let raw: (index: number) => number
  let block = 0
  let head = 0

  if (blocks) {
    block = blocks.length - 1
    const peaks = blocks[block].map((row) => Math.max(...row.slice(1)))
    head = peaks.indexOf(Math.max(...peaks))
    // Позиция 0 — сам [USR], события — с 1.
    raw = (index) => blocks[block][head][1 + index]
  } else {
    let seed = 97
    const noise = timeline.map(() => {
      seed = (seed * 1103515245 + 12345) % 2147483648
      return seed / 2147483648
    })
    raw = (index) => {
      const event = timeline[index]
      return (0.25 + Math.exp(-event.time_log / 45)) * (event.type === 'salary_credit' ? 2.2 : 1) * (0.6 + 0.8 * noise[index])
    }
  }

  const heaviest = blocks
    ? timeline
        .map((_, index) => index)
        .sort((a, b) => raw(b) - raw(a))
        .slice(0, SHOWN / 2)
    : []
  const picks = new Set<number>([hero.index, ...heaviest])

  const rest = timeline.map((_, index) => index).filter((index) => !picks.has(index))
  const free = Math.min(SHOWN - picks.size, rest.length)
  for (let j = 0; j < free; j++) picks.add(rest[Math.round((j * (rest.length - 1)) / Math.max(1, free - 1))])

  const indices = [...picks].sort((a, b) => a - b)
  const max = Math.max(...indices.map(raw))

  return {
    events: indices.map((index) => ({ index, weight: raw(index) / max })),
    heaviest,
    real: blocks !== undefined,
    block,
    head,
    max,
  }
}

export interface Candidate {
  value_id: number
  // Диапазон суммы, если токен — корзина скрытого ключа; иначе имя токена.
  bucket: (BucketRange & { value_id: number }) | null
  name: string
  p: number
  target: boolean
  top: boolean
}

export function bucketNumber(name: string): number {
  return Number(name.split('_').pop())
}

// С чекпойнтом: top-5 по всему словарю и цель, по убыванию
// вероятности. Без него — пять диапазонов вокруг цели со
// схематичными вероятностями.
export function candidates(): Candidate[] {
  const mlm = demo.model?.mlm
  const scale = demo.client.mlm_candidates
  const byId = new Map(scale.buckets.map((bucket) => [bucket.value_id, bucket]))

  if (mlm) {
    const rows: Candidate[] = mlm.top5.map((item, index) => ({
      value_id: item.value_id,
      bucket: byId.get(item.value_id) ?? null,
      name: item.name,
      p: item.p,
      target: item.value_id === mlm.target,
      top: index === 0,
    }))
    if (!rows.some((row) => row.target)) {
      rows.push({
        value_id: mlm.target,
        bucket: byId.get(mlm.target) ?? null,
        name: byId.get(mlm.target)?.name ?? String(mlm.target),
        p: mlm.target_probability,
        target: true,
        top: false,
      })
    }
    return rows
  }

  const all = [...scale.buckets].sort((a, b) => bucketNumber(a.name) - bucketNumber(b.name))
  const at = all.findIndex((bucket) => bucket.value_id === scale.target)
  const from = Math.max(0, Math.min(all.length - 5, at - 2))
  const shown = all.slice(from, from + 5)
  const weights = shown.map((bucket) => Math.exp(-Math.abs(bucketNumber(bucket.name) - bucketNumber(all[at].name)) * 1.25))
  const total = weights.reduce((a, b) => a + b, 0) / 0.86
  return shown.map((bucket, index) => ({
    value_id: bucket.value_id,
    bucket,
    name: bucket.name,
    p: weights[index] / total,
    target: bucket.value_id === scale.target,
    top: false,
  }))
}
