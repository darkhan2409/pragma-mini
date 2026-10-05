import { ANCHOR, type RegionId, type Vec3 } from './layout'

// ============================================================
// КАДРЫ КАМЕРЫ
// ============================================================
//
// Кадр считается по габаритам содержимого региона: всё
// помещается по ширине и в верхние две трети экрана (нижнюю
// четверть занимает подпись). Камера смотрит прямо, без наклона:
// вертикали остаются вертикалями, карточки — прямоугольниками.
// ============================================================

export interface Shot {
  position: Vec3
  target: Vec3
}

export const FOV = 38
export const TAN = Math.tan(((FOV / 2) * Math.PI) / 180)

// Самый узкий экран, под который считается кадр (16:10).
const ASPECT = 1.6

// Доли экрана под содержимое: по ширине и по высоте между верхней
// панелью и подписью; центр содержимого — на 0.4 высоты сверху.
const WIDTH_SHARE = 0.92
const HEIGHT_SHARE = 0.64
const RAISE = 0.1

// Габариты содержимого региона в его координатах: центр и размер.
type Box = [cx: number, cy: number, width: number, height: number]

const BOX: Partial<Record<RegionId, Box>> = {
  client: [-0.5, 0.5, 29.6, 11.6],
  event: [1.4, -0.1, 24.2, 13.8],
  tokens: [-1.2, 0.3, 28.8, 12.0],
  embedding: [1.8, 0.6, 24.0, 12.2],
  eventEncoder: [0.3, 0.7, 27.6, 12.0],
  calendar: [-1.9, 1.3, 26.2, 10.2],
  profile: [-3.6, 0.4, 21.8, 12.2],
  history: [-0.4, 2.2, 29.6, 8.2],
  rope: [-0.7, 1.0, 28.6, 11.0],
  historyEncoder: [-0.3, 1.0, 29.4, 11.0],
  mlm: [0.2, 0.4, 30.8, 11.2],
  loop: [-0.1, 0.4, 29.8, 12.2],
  downstream: [0.3, 0.4, 30.6, 12.8],
  clientEmbedding: [0, 0.9, 17, 15.8],
  lrTraining: [0, 0.5, 30.4, 12.4],
  catboostTraining: [0, 0.5, 30.4, 12.4],
}

export function fit(region: RegionId): Shot {
  const [x, y, z] = ANCHOR[region]
  const [cx, cy, width, height] = BOX[region]!
  const visible = Math.max(height / HEIGHT_SHARE, width / (WIDTH_SHARE * ASPECT))
  const distance = visible / (2 * TAN)
  const targetY = y + cy - RAISE * visible
  return {
    position: [x + cx, targetY, z + distance],
    target: [x + cx, targetY, z],
  }
}

export const SHOTS = {
  client: fit('client'),
  clientClose: (() => {
    const shot = fit('client')
    return { position: [shot.position[0], shot.position[1], shot.position[2] * 0.82], target: shot.target } as Shot
  })(),
  event: fit('event'),
  tokens: fit('tokens'),
  embedding: fit('embedding'),
  eventEncoder: fit('eventEncoder'),
  calendar: fit('calendar'),
  profile: fit('profile'),
  history: fit('history'),
  rope: fit('rope'),
  historyEncoder: fit('historyEncoder'),
  mlm: fit('mlm'),
  loop: fit('loop'),
  downstream: fit('downstream'),
  lrTraining: fit('lrTraining'),
  catboostTraining: fit('catboostTraining'),
  finalClose: fit('clientEmbedding'),
  // Общие планы: цепочка от таблицы до головы и до [USR].
  backprop: { position: [146, 4, 116], target: [146, 4, 0] } as Shot,
  final: { position: [156, -1, 128], target: [156, -1, 0] } as Shot,
  // Весь мир: от клиента до обучения голов.
  overview: { position: [140, 0, 300], target: [140, 0, 0] } as Shot,
} satisfies Record<string, Shot>

export type ShotId = keyof typeof SHOTS
