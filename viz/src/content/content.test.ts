import { readFileSync, readdirSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { describe, expect, test } from 'vitest'
import { demo } from '../data/demo'
import { codepoints } from '../test/cmap'
import { ANCHOR, REGIONS } from './layout'
import { PARTS } from './parts'
import { SHOTS } from './shots'
import { STEPS, shotOf } from './steps'
import { CAUSES, EXPERIMENTS, SOURCE } from './wave'

// ============================================================
// Содержание презентации: 18 шагов, у каждого есть кадр или
// экран, регионы существуют, иллюстрации помечены; эксперименты —
// дословно из README аудита; всё, что пишет сцена, есть в её
// шрифтах (иначе troika пошла бы в CDN).
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

  test('2D-экраны — шаги 14, 17 и 18: обучение, главный вывод, причины', () => {
    expect(STEPS.map((step, index) => (step.screen ? index + 1 : null)).filter(Boolean)).toEqual([14, 17, 18])
    expect(STEPS.filter((step) => step.screen).map((step) => step.screen)).toEqual(['dashboard', 'insight', 'causes'])
  })

  test('у шага причин по биту на причину, и каждая проверка волны 4 — из списка экспериментов', () => {
    const causes = STEPS.find((step) => step.screen === 'causes')!
    expect(causes.beats).toEqual(CAUSES.map((cause) => cause.title))
    const ids = EXPERIMENTS.map((item) => item.id)
    for (const cause of CAUSES) {
      expect(cause.checks.length).toBeGreaterThan(0)
      for (const check of cause.checks) if (check.id) expect(ids).toContain(check.id)
    }
  })

  test('регионы шагов существуют, и у каждого 3D-шага есть регион в фокусе', () => {
    for (const step of STEPS) {
      for (const region of Object.keys(step.regions)) expect(REGIONS).toContain(region)
      if (!step.screen) expect(Object.values(step.regions)).toContain('focus')
    }
  })

  test('числа обученной модели помечены там, где они есть в экспорте', () => {
    const real = Object.fromEntries(STEPS.map((step) => [step.id, step.checkpoint ?? []]))
    expect(real.historyEncoder).toEqual(demo.model ? ['attention'] : [])
    expect(real.mlm).toEqual(demo.model ? ['topk'] : [])
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

describe('волна 4', () => {
  const readme = readFileSync(join(ROOT, SOURCE), 'utf8')

  test('эксперименты проверок — из README аудита', () => {
    for (const item of EXPERIMENTS) expect(readme).toContain(`| ${item.id} |`)
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
