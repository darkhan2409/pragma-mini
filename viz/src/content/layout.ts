// ============================================================
// РАСКЛАДКА МИРА
// ============================================================
//
// Один мир, регионы вдоль X в порядке pipeline. Параллельные
// дорожки: календарь под энкодером события, анкета над историей,
// обучение («верхняя палуба») над History Encoder. Позиции
// регионов — только здесь.
// ============================================================

export type Vec3 = [number, number, number]

export const REGIONS = [
  'client',
  'event',
  'tokens',
  'embedding',
  'eventEncoder',
  'calendar',
  'profile',
  'history',
  'rope',
  'historyEncoder',
  'mlm',
  'backprop',
  'loop',
  'clientEmbedding',
  'downstream',
] as const

export type RegionId = (typeof REGIONS)[number]

export const ANCHOR: Record<RegionId, Vec3> = {
  client: [0, 0, 0],
  event: [32, 0, 0],
  tokens: [64, 0, 0],
  embedding: [96, 0, 0],
  eventEncoder: [128, 0, 0],
  calendar: [128, -15, 0],
  profile: [160, 15, 0],
  history: [160, 0, 0],
  rope: [160, -15, 0],
  historyEncoder: [192, 0, 0],
  mlm: [192, 16, 0],
  backprop: [150, 6, 0],
  loop: [224, 16, 0],
  clientEmbedding: [214, 0, 0],
  downstream: [240, 0, 0],
}

// Заголовки регионов на общем плане (Rail).
export const REGION_TITLE: Partial<Record<RegionId, string>> = {
  client: 'Клиент',
  event: 'Событие',
  tokens: 'Токены',
  embedding: 'Таблица эмбеддингов',
  eventEncoder: 'Event Encoder',
  calendar: 'Календарь',
  profile: 'Profile Encoder',
  history: 'История',
  rope: 'TimeRoPE',
  historyEncoder: 'History Encoder',
  mlm: 'MLM head',
  loop: 'Цикл обучения',
  clientEmbedding: 'Client Embedding',
  downstream: 'Задачи',
}

// Путь данных для рельса: основная линия и притоки.
export const RAIL_MAIN: RegionId[] = [
  'client',
  'event',
  'tokens',
  'embedding',
  'eventEncoder',
  'history',
  'historyEncoder',
  'clientEmbedding',
  'downstream',
]

export const RAIL_BRANCHES: [RegionId, RegionId][] = [
  ['eventEncoder', 'calendar'],
  ['calendar', 'history'],
  ['embedding', 'profile'],
  ['profile', 'history'],
  ['history', 'rope'],
  ['historyEncoder', 'mlm'],
  ['mlm', 'loop'],
]

export function at(region: RegionId, dx = 0, dy = 0, dz = 0): Vec3 {
  const [x, y, z] = ANCHOR[region]
  return [x + dx, y + dy, z + dz]
}
