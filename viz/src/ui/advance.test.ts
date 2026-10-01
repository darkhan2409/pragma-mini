import { describe, expect, test } from 'vitest'
import { CONTROLS, shouldAdvance } from './advance'

// Цель клика: closest находит предка по селектору, как в DOM.
function target(...ancestors: string[]) {
  const selectors = CONTROLS.split(',').map((item) => item.trim())
  return {
    closest: (selector: string) => {
      expect(selector).toBe(CONTROLS)
      return ancestors.some((tag) => selectors.includes(tag)) ? {} : null
    },
  }
}

describe('клик по 2D-экрану', () => {
  test('по тексту и карточке — дальше', () => {
    expect(shouldAdvance(target('div', 'p'), '')).toBe(true)
    expect(shouldAdvance(target('td', 'table', 'div'), '')).toBe(true)
  })

  test('по элементам управления — нет', () => {
    for (const control of ['button', 'a', 'input', 'select', 'textarea', 'label', '[data-no-advance]']) {
      expect(shouldAdvance(target('span', control, 'div'), '')).toBe(false)
    }
  })

  test('при выделенном тексте — нет', () => {
    expect(shouldAdvance(target('p'), 'val loss')).toBe(false)
    expect(shouldAdvance(target('p'), '   ')).toBe(true)
  })

  test('без цели — дальше', () => {
    expect(shouldAdvance(null, '')).toBe(true)
  })
})
