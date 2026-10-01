import { readFileSync, readdirSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { describe, expect, test } from 'vitest'
import { demo } from '../data/demo'
import { codepoints } from '../test/cmap'
import { ANCHOR, REGIONS } from './layout'
import { PARTS } from './parts'
import { SHOTS } from './shots'
import { STEPS, shotOf } from './steps'
import { ACCEPTED_BASELINE, EXPERIMENTS, SOURCE, WAVE1 } from './wave'

// ============================================================
// Содержание презентации: 18 шагов, у каждого есть кадр или
// экран, регионы существуют, иллюстрации помечены; числа волны 1
// и эксперименты — дословно из README аудита; всё, что пишет
// сцена, есть в её шрифтах (иначе troika пошла бы в CDN).
// ============================================================

const VIZ = resolve(__dirname, '..', '..')
const ROOT = resolve(VIZ, '..')

describe('шаги', () => {
  test('18 шагов с уникальными id, у каждого — кадр или 2D-экран', () => {
    expect(STEPS).toHaveLength(18)
    expect(new Set(STEPS.map((step) => step.id)).size).toBe(18)
    for (const step of STEPS) {
      expect(step.beats.length).toBeGreaterThan(0)
      if (step.screen) continue
      for (let beat = 0; beat < step.beats.length; beat++) {
        const shot = shotOf(step, beat)
        expect(shot, `${step.id}: кадр бита ${beat}`).toBeDefined()
        expect(SHOTS[shot!], `${step.id}: кадр ${shot}`).toBeDefined()
      }
    }
  })

  test('2D-экраны — шаги 14, 17 и 18', () => {
    expect(STEPS.map((step, index) => (step.screen ? index + 1 : null)).filter(Boolean)).toEqual([14, 17, 18])
  })

  test('регионы шагов существуют, и у каждого 3D-шага есть регион в фокусе', () => {
    for (const step of STEPS) {
      for (const region of Object.keys(step.regions)) expect(REGIONS).toContain(region)
      if (!step.screen) expect(Object.values(step.regions)).toContain('focus')
    }
  })

  test('иллюстрации помечены там, где числа не из модели', () => {
    const marked = Object.fromEntries(STEPS.map((step) => [step.id, step.illustrative ?? []]))
    expect(marked.embedding).toContain('vectors')
    expect(marked.eventEncoder).toContain('attention')
    expect(marked.rope).toContain('qk')
    // Веса внимания и вероятности — схема, пока в экспорте нет
    // чисел обученной модели (export_demo.py --checkpoint).
    expect(marked.historyEncoder.includes('attention')).toBe(demo.model === null)
    expect(marked.mlm.includes('topk')).toBe(demo.model === null)
    expect(marked.mlm).toContain('vectors')
    const real = Object.fromEntries(STEPS.map((step) => [step.id, step.checkpoint ?? []]))
    expect(real.historyEncoder).toEqual(demo.model ? ['attention'] : [])
    expect(real.mlm).toEqual(demo.model ? ['topk'] : [])
    expect(marked.backprop).toContain('gradients')
  })

  test('детали EXPLORE ведут на существующие шаги', () => {
    for (const [id, part] of Object.entries(PARTS)) {
      expect(STEPS.some((step) => step.id === part.step), `${id} → ${part.step}`).toBe(true)
    }
  })

  test('у каждого региона есть якорь', () => {
    for (const region of REGIONS) expect(ANCHOR[region]).toHaveLength(3)
  })
})

describe('волны аудита', () => {
  const readme = readFileSync(join(ROOT, SOURCE), 'utf8')

  test('каждое число волны 1 есть в README аудита', () => {
    for (const row of WAVE1.rows) {
      expect(readme).toContain(row.set === 'CatBoost' ? 'CatBoost' : `\`${row.set}\``)
      for (const value of row.values) if (value !== null) expect(readme).toContain(value.toFixed(3))
    }
  })

  test('эксперименты и принятое в эталон — из README аудита', () => {
    for (const item of EXPERIMENTS) expect(readme).toContain(`| ${item.id} |`)
    for (const item of ACCEPTED_BASELINE) expect(readme).toContain(item.id)
  })
})

describe('шрифты сцены', () => {
  const inter = codepoints(join(VIZ, 'public/fonts/Inter-Variable.ttf'))
  const mono = codepoints(join(VIZ, 'public/fonts/JetBrainsMono-Regular.ttf'))

  // Символ, которого нет в одном из шрифтов, допустим только там,
  // где пишет второй: ␠ — в моноширинных фишках кусков BPE, ₸ —
  // только в подписях Inter.
  const ONLY = new Map([
    ['␠', mono],
    ['₸', inter],
  ])

  const sources = [
    ...readdirSync(join(VIZ, 'src/three/regions')).map((name) => join(VIZ, 'src/three/regions', name)),
    join(VIZ, 'src/three/primitives.tsx'),
    join(VIZ, 'src/three/Stage.tsx'),
    join(VIZ, 'src/content/layout.ts'),
    join(VIZ, 'src/content/steps.ts'),
    join(VIZ, 'src/data/demo.ts'),
    join(VIZ, 'src/data/pragma_demo.json'),
  ]

  test('все символы сцены и данных есть в Inter и JetBrains Mono', () => {
    const text = sources.map((path) => readFileSync(path, 'utf8')).join('')
    const missing = [...new Set([...text])].filter((char) => {
      const code = char.codePointAt(0)!
      if (code <= 126) return false
      const only = ONLY.get(char)
      return only ? !only.has(code) : !(inter.has(code) && mono.has(code))
    })
    expect(missing).toEqual([])
  })

  test('₸ не попадает в моноширинные подписи', () => {
    for (const path of sources.filter((item) => item.endsWith('.tsx'))) {
      for (const line of readFileSync(path, 'utf8').split('\n')) {
        if (line.includes('₸')) expect(line, path).not.toMatch(/mono/)
      }
    }
    expect(JSON.stringify(demo)).not.toContain('₸')
  })
})
