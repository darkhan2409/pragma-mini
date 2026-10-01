// ============================================================
// ДИЗАЙН-ТОКЕНЫ
// ============================================================
//
// Один источник цветов для DOM (CSS-переменные в styles.css
// повторяют эти значения) и для three.js. Акцентов четыре, и за
// шаг используется не больше двух:
//
//   cyan    токены, векторы событий, поток данных
//   amber   [USR], Client Embedding, даунстрим
//   violet  время: календарь, TimeRoPE, давность
//   rose    [MASK], loss, градиенты
// ============================================================

export const color = {
  bg: '#07090D',
  surface: '#0D1117',
  surface2: '#131A23',
  line: '#1E2733',
  text: '#E6EBF1',
  text2: '#9AA5B1',
  muted: '#5B6672',
  cyan: '#46C8E0',
  amber: '#F0B44C',
  violet: '#9B8CFF',
  rose: '#FF5C7A',
  green: '#5CCB8A',
} as const

// Приглушённые тона источников выгрузки: цвет на оси времени
// различает потоки, но не спорит с акцентами.
export const sourceColor: Record<string, string> = {
  transactions: '#5E8FA8',
  loans: '#A88A5E',
  product_events: '#8A7FB8',
  applications: '#B87F9E',
  app_screens: '#4E6E7E',
  app_operations: '#5F8578',
  communications: '#7A7F8C',
  banners: '#6B6478',
  antifraud: '#B86B6B',
  support: '#7F9468',
  profile: '#9A8F6A',
}

export const FONT = {
  sans: '/fonts/Inter-Variable.ttf',
  mono: '/fonts/JetBrainsMono-Regular.ttf',
} as const

// Состояния регионов сцены и их видимость.
export const FADE = { focus: 1, context: 0.3, hidden: 0 } as const
