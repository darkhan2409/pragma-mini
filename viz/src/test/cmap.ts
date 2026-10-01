import { readFileSync } from 'node:fs'

// ============================================================
// ГЛИФЫ ШРИФТА ПО ТАБЛИЦЕ cmap (TrueType)
// ============================================================
//
// troika-three-text за недостающими глифами идёт в CDN; тест
// проверяет, что всё, что сцена пишет, есть в своих шрифтах.
// Разбираются форматы cmap 4 и 12 — их используют Inter и
// JetBrains Mono.
// ============================================================

export function codepoints(path: string): Set<number> {
  const data = readFileSync(path)
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
  const tables = view.getUint16(4)
  let cmap = -1
  for (let i = 0; i < tables; i++) {
    const record = 12 + i * 16
    const tag = String.fromCharCode(...[0, 1, 2, 3].map((k) => view.getUint8(record + k)))
    if (tag === 'cmap') cmap = view.getUint32(record + 8)
  }
  if (cmap < 0) throw new Error(`${path}: нет таблицы cmap`)

  const found = new Set<number>()
  const count = view.getUint16(cmap + 2)

  for (let i = 0; i < count; i++) {
    const offset = cmap + view.getUint32(cmap + 4 + i * 8 + 4)
    const format = view.getUint16(offset)

    if (format === 4) {
      const segments = view.getUint16(offset + 6) / 2
      const ends = offset + 14
      const starts = ends + segments * 2 + 2
      for (let s = 0; s < segments; s++) {
        const end = view.getUint16(ends + s * 2)
        const start = view.getUint16(starts + s * 2)
        for (let c = start; c <= end && c !== 0xffff; c++) found.add(c)
      }
    } else if (format === 12) {
      const groups = view.getUint32(offset + 12)
      for (let g = 0; g < groups; g++) {
        const base = offset + 16 + g * 12
        const start = view.getUint32(base)
        const end = view.getUint32(base + 4)
        for (let c = start; c <= end; c++) found.add(c)
      }
    }
  }

  return found
}
