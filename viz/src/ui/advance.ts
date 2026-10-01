// ============================================================
// КЛИК ПО 2D-ЭКРАНУ
// ============================================================
//
// В PRESENT клик по экрану листает дальше, как клик по сцене. Не
// листают элементы управления (кнопки, ссылки, поля, выбор) и клик,
// которым выделили текст: ими пользуются, а не переходят.
// ============================================================

export const CONTROLS = 'button, a, input, select, textarea, label, [data-no-advance]'

export interface ClickTarget {
  closest(selector: string): unknown
}

export function shouldAdvance(target: ClickTarget | null, selection: string): boolean {
  if (selection.trim()) return false
  return !target?.closest(CONTROLS)
}
