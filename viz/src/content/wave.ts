// ============================================================
// ВОЛНА 4
// ============================================================
//
// Источник — audit/2026-09-28-project/wave4/README.md, тест сверки
// (content.test.ts) проверяет эксперименты по нему. Результатов
// волны 4 здесь нет: статусы приходят из каталогов прогонов,
// решения — из data/wave4_decisions.json, числа проб — из экспорта.
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
  'решает только val: Δ PR-AUC и ROC-AUC сценариев usr и catboost_plus_usr на churn_active90 к текущему лучшему',
  'MLM val loss — вторичный критерий',
  'test смотрится один раз, на финальном прогоне победителя',
]
