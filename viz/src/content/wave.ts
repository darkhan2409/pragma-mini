import { demo, int } from '../data/demo'

// ============================================================
// ПОЧЕМУ ТАК: ВОЗМОЖНЫЕ ПРИЧИНЫ И ПРОВЕРКИ
// ============================================================
//
// Финальный шаг: почему [USR] после MLM далеко ниже handcrafted-
// признаков и к ним ничего не добавляет. Причины — гипотезы, у
// каждой — что проверить: эксперимент волны 4 или работа вне неё.
//
// Эксперименты — из audit/2026-09-28-project/wave4/README.md, тест
// сверки (content.test.ts) проверяет каждый id по нему. Статусы
// приходят из каталогов прогонов, решения — из
// data/wave4_decisions.json. Числа модели — только из экспорта.
// ============================================================

export const SOURCE = 'audit/2026-09-28-project/wave4/README.md'

export interface Experiment {
  id: string
  title: string
  enable: string
}

// Эксперименты, на которые ссылаются причины.
export const EXPERIMENTS: Experiment[] = [
  { id: 'E2', title: 'глубина: событие 2, история 5', enable: 'event-layers2.json + history-layers5.json' },
  { id: 'E4', title: 'dropout 0', enable: '*-dropout0.json' },
]

export interface Check {
  // id эксперимента волны 4; без него — проверка вне волны 4.
  id?: string
  text: string
}

export interface Cause {
  title: string
  why: string
  checks: Check[]
}

const encoders = demo.architecture.encoders
const clients = int(demo.batching.clients)
const epochs = (demo.run?.epochs ?? []).map((item) => item.val_loss).filter((value): value is number => value !== null)
const falling = epochs.length >= 2 && epochs[epochs.length - 1] < epochs[epochs.length - 2]

export const CAUSES: Cause[] = [
  {
    title: 'Модель недоучена',
    why: falling
      ? `Ошибка ещё падала на последней эпохе (${epochs[epochs.length - 2].toFixed(3)} → ${epochs[epochs.length - 1].toFixed(3)}).`
      : 'Обучение остановлено по счётчику эпох.',
    checks: [{ text: 'Доучить B0 ещё несколько эпох.' }, { id: 'E4', text: 'Обучить без dropout.' }],
  },
  {
    title: 'Вектор заморожен, голова линейная',
    why: 'Вектор берём как есть, сверху — простая регрессия. В статье PRAGMA его ещё доучивают под задачу (LoRA).',
    checks: [{ text: 'Доучить модель под отток (LoRA).' }],
  },
  {
    title: 'Глубина не там',
    why: `На событие — ${encoders.event.blocks} слоёв, на всю историю клиента — ${encoders.history.blocks}.`,
    checks: [{ id: 'E2', text: 'Больше слоёв на историю, меньше на событие.' }],
  },
  {
    title: 'Отток в синтетике задают счётчики',
    why: 'Уход в генераторе — пауза в действиях. Её предсказывают частота и давность действий, а новые паузы случайны: порядок событий для метки не важен.',
    checks: [
      { text: 'CatBoost только на давности и счётчиках против всех признаков.' },
      { text: 'Задача, где нужна именно история событий.' },
    ],
  },
  {
    title: 'Мало клиентов',
    why: `Модель училась на ${clients} клиентах, в статье PRAGMA — на 26 млн.`,
    checks: [{ text: 'Обучить B0 на 25%, 50% и 100% клиентов и сравнить.' }],
  },
]
