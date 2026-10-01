// ============================================================
// ВОЛНЫ АУДИТА
// ============================================================
//
// Источник — audit/2026-09-28-project/wave4/README.md. Числа
// волны 1 переписаны оттуда дословно, и тест сверки
// (content.test.ts) проверяет, что каждое из них там есть.
// Результатов волны 4 здесь нет: статусы приходят из каталогов
// прогонов, решения — из data/wave4_decisions.json.
// ============================================================

export const SOURCE = 'audit/2026-09-28-project/wave4/README.md'

export interface Experiment {
  id: string
  title: string
  enable: string
  rebuild: string
  hypothesis: string
}

export const EXPERIMENTS: Experiment[] = [
  { id: 'B0', title: 'эталон', enable: 'без конфигов', rebuild: '—', hypothesis: 'точка отсчёта для всех сравнений' },
  { id: 'N', title: 'шумовой пол', enable: 'train-seed43.json', rebuild: '—', hypothesis: 'какая разница между прогонами — просто шум' },
  {
    id: 'E1',
    title: '[USR]: доли типов событий за 7/30/90 дней',
    enable: 'train-usr-aux.json',
    rebuild: '—',
    hypothesis: 'своя цель у [USR] сделает вектор клиента полезнее',
  },
  {
    id: 'E2',
    title: 'глубина: событие 2, история 5',
    enable: 'event-layers2.json + history-layers5.json',
    rebuild: '07_backbone',
    hypothesis: 'глубина нужнее в истории, чем внутри события',
  },
  { id: 'E3', title: 'softmax по кандидатам ключа', enable: 'train-restricted-softmax.json', rebuild: '—', hypothesis: 'не тратить вероятность на чужие ключи' },
  { id: 'E4', title: 'dropout 0', enable: '*-dropout0.json', rebuild: '07_backbone', hypothesis: 'на синтетике регуляризация мешает' },
  { id: 'E5', title: 'перестановка групп строк по эпохам', enable: 'train-shuffle.json', rebuild: '—', hypothesis: 'порядок клиентов в эпохе влияет на обучение' },
  { id: 'E6', title: 'закрыть ключи события под маской event', enable: 'train-hide-event-keys.json', rebuild: '—', hypothesis: 'ключи не подсказывают тип события' },
  {
    id: 'L',
    title: 'прежний словарь (ByteLevel, общая шкала суммы)',
    enable: 'tokenizer-legacy.json',
    rebuild: '03–07',
    hypothesis: 'проверка: новый словарь не хуже прежнего',
  },
  { id: 'LT', title: 'прежний отсчёт времени: от последнего события', enable: 'dataset-last-event.json', rebuild: '05', hypothesis: 'проверка: время от T не хуже' },
]

// Принятое в эталон без A/B решением владельца.
export const ACCEPTED_BASELINE = [
  { id: 'E7', text: 'время событий от cutoff T, а не от последнего события' },
  { id: 'E8', text: 'BPE characters' },
  { id: 'E9', text: 'шкалы денег: сумма по направлению, минус остатка своими квантилями' },
  { id: 'E10', text: 'product_id и previous_product_id — категории (2026-09-30)' },
]

export const MEASUREMENT_FIXES = [
  'порча контекста выбранного ключа в [UNK], вероятность 0.5',
  'взвешенное по информативности value-маскирование (0.05–0.35)',
]

export const PROTOCOL = [
  'решает только val: Δ PR-AUC и ROC-AUC наборов usr+recency и readouts+recency на churn_active90',
  'MLM val loss — вторичный критерий',
  'test смотрится один раз, на финальном прогоне победителя',
]

// Волна 1: test PR-AUC, 10-эпохная модель, прежние данные и код.
export const WAVE1 = {
  caption: 'Волна 1 — test PR-AUC; 10-эпохная модель на прежних данных и коде, до волны 4',
  columns: ['churn'],
  rows: [
    { set: 'usr', values: [0.256] },
    { set: 'init:usr', values: [0.301] },
    { set: 'readouts+recency', values: [0.422] },
    { set: 'init:readouts+recency', values: [0.435] },
    { set: 'counts', values: [0.514] },
    { set: 'CatBoost', values: [0.618] },
  ] as { set: string; values: (number | null)[] }[],
  conclusion: 'Обученный [USR] не лучше необученного, и простые счётчики обгоняют все векторы.',
}
