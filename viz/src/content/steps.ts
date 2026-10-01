import { demo, hero, int, num } from '../data/demo'
import type { RegionId } from './layout'
import type { ShotId } from './shots'

// ============================================================
// ШАГИ ПРЕЗЕНТАЦИИ
// ============================================================
//
// Единственный источник содержания PRESENT: заголовок, подпись,
// биты (подшаги по стрелке), какие регионы в фокусе, какой кадр,
// какой 2D-экран и что на шаге — иллюстрация, а не модель.
// Числа в подписях — из экспорта (data/demo.ts).
// ============================================================

export type RegionState = 'focus' | 'context' | 'hidden'

export type Illustration = 'vectors' | 'attention' | 'qk' | 'topk' | 'gradients'

export type Screen = 'dashboard' | 'wave4' | 'insight'

export interface Step {
  id: string
  title: string
  caption: string
  beats: string[]
  regions: Partial<Record<RegionId, RegionState>>
  shot?: ShotId | ShotId[]
  screen?: Screen
  illustrative?: Illustration[]
  // Что на шаге — числа обученной модели (export_demo.py --checkpoint).
  checkpoint?: Illustration[]
  part?: string
}

const client = demo.client
const arch = demo.architecture
const training = demo.training
const masking = demo.masking
const run = demo.run
const amount = hero.raw.transaction_amount as number
const merchant = String(hero.raw.merchant_name)
const perEpoch = demo.batching.micro_batches
const epochs = Number(run?.plan?.epochs ?? 5)
const totalSteps = run?.total_steps ?? perEpoch * epochs

export const STEPS: Step[] = [
  {
    id: 'client',
    title: 'Клиент',
    caption: `Один клиент банка — ${int(client.n_events)} событий за два года. Каждое событие — точка на оси времени до cutoff T.`,
    beats: ['клиент', 'события по источникам', 'cutoff T'],
    regions: { client: 'focus' },
    shot: ['clientClose', 'client', 'client'],
    part: 'client',
  },
  {
    id: 'event',
    title: 'Одно событие',
    caption: `Покупка в ${merchant} на ${int(amount)} ₸. Событие — набор смысловых пар ключ = значение; время идёт отдельным каналом.`,
    beats: ['событие выходит вперёд', 'поля события'],
    regions: { event: 'focus' },
    shot: 'event',
    part: 'event',
  },
  {
    id: 'tokens',
    title: 'Токенизация',
    caption:
      'Каждая пара — номера одного словаря: ключ — свой токен, значение — категория, диапазон суммы или куски BPE с позициями 0, 1, …',
    beats: ['пары', 'токены значений', 'номера и позиции'],
    regions: { tokens: 'focus' },
    shot: 'tokens',
    part: 'tokens',
  },
  {
    id: 'embedding',
    title: 'Входной эмбеддинг',
    caption: `Одна таблица ${int(demo.vocabulary.size)} × ${arch.dim} на все токены. Вектор токена — строки ключа и значения, умноженные на √d, плюс позиция куска.`,
    beats: ['таблица по видам токенов', 'выбор строк', 'векторы [128]'],
    regions: { embedding: 'focus' },
    shot: 'embedding',
    illustrative: ['vectors'],
    part: 'embedding',
  },
  {
    id: 'eventEncoder',
    title: 'Event Encoder',
    caption: `${arch.encoders.event.blocks} блоков трансформера над токенами одного события; внимание не выходит за его границы. На выходе — вектор события на месте [EVT].`,
    beats: ['события по отдельности', 'блоки: внимание, FFN, residual', '[EVT] → вектор события'],
    regions: { eventEncoder: 'focus' },
    shot: 'eventEncoder',
    illustrative: ['attention', 'vectors'],
    part: 'eventEncoder',
  },
  {
    id: 'calendar',
    title: 'Календарь',
    caption:
      'Час, день недели и день месяца — 6 чисел sin/cos → MLP → [128]. Прибавляется к вектору события после Event Encoder, а не к токенам.',
    beats: ['время события на окружностях', 'MLP и сумма'],
    regions: { calendar: 'focus' },
    shot: 'calendar',
    illustrative: ['vectors'],
    part: 'calendar',
  },
  {
    id: 'profile',
    title: 'Profile Encoder',
    caption: `Анкета: [USR], ${demo.dataset.profile_fields.length} полей и вехи с давностью до T. Один блок трансформера → вектор анкеты [128] на месте [USR].`,
    beats: ['токены анкеты', 'блок и вектор анкеты'],
    regions: { profile: 'focus' },
    shot: 'profile',
    illustrative: ['vectors'],
    part: 'profile',
  },
  {
    id: 'history',
    title: 'История клиента',
    caption:
      'Вектор анкеты и датированные векторы событий — одна последовательность до cutoff T. Анкета и события встречаются здесь впервые.',
    beats: ['последовательность по времени', 'позиция — давность до T'],
    regions: { history: 'focus' },
    shot: 'history',
    part: 'history',
  },
  {
    id: 'rope',
    title: 'TimeRoPE',
    caption:
      'Позиция — давность до T: 8·log1p(Δt/8). Она поворачивает Q и K, V не трогает: внимание зависит от разницы во времени.',
    beats: ['Q, K, V', 'поворот Q и K'],
    regions: { rope: 'focus' },
    shot: 'rope',
    illustrative: ['qk'],
    part: 'rope',
  },
  {
    id: 'historyEncoder',
    title: 'History Encoder',
    caption: `${arch.encoders.history.blocks} блока × ${arch.encoders.history.config.heads} головы, внимание в обе стороны. [USR] собирает всю историю клиента — это и есть Client Embedding.`,
    beats: ['блоки', 'внимание [USR] к событиям', 'выходы'],
    regions: { historyEncoder: 'focus' },
    shot: 'historyEncoder',
    illustrative: demo.model ? [] : ['attention'],
    checkpoint: demo.model ? ['attention'] : [],
    part: 'historyEncoder',
  },
  {
    id: 'mlm',
    title: 'MLM: угадать скрытое',
    caption: `Значение суммы прячем под [MASK]. Голова берёт три вектора → concat 384 → Linear → 128 → логиты той же таблицей эмбеддингов.`,
    beats: ['маска', 'три входа головы', 'предсказание'],
    regions: { mlm: 'focus' },
    shot: 'mlm',
    illustrative: demo.model ? ['vectors'] : ['topk', 'vectors'],
    checkpoint: demo.model ? ['topk'] : [],
    part: 'mlm',
  },
  {
    id: 'backprop',
    title: 'Loss и backward',
    caption: `Cross-entropy (сглаживание ${num(Number(training.label_smoothing), 1)}) → градиенты идут назад через голову, историю, энкодеры и таблицу → clip ${num(Number(training.max_grad_norm), 1)} → шаг AdamW.`,
    beats: ['loss', 'градиенты назад и шаг'],
    regions: {
      backprop: 'focus',
      mlm: 'context',
      historyEncoder: 'context',
      history: 'context',
      profile: 'context',
      eventEncoder: 'context',
      embedding: 'context',
    },
    shot: 'backprop',
    illustrative: ['gradients'],
    part: 'backprop',
  },
  {
    id: 'loop',
    title: 'Цикл обучения',
    caption: `Клиенты упакованы в micro-batch до ${int(Number(training.token_budget))} токенов без паддинга: forward → loss → backward → шаг. ${int(perEpoch)} шагов — одна эпоха, всего ${int(totalSteps)}.`,
    beats: ['один шаг', 'эпохи'],
    regions: { loop: 'focus' },
    shot: 'loop',
    part: 'loop',
  },
  {
    id: 'dashboard',
    title: 'Обучение',
    caption: 'Живая телеметрия прогона: читается из telemetry.jsonl каталога прогона.',
    beats: ['дашборд'],
    regions: {},
    screen: 'dashboard',
  },
  {
    id: 'final',
    title: 'Client Embedding',
    caption: `MLM-голова нужна только для обучения. После него остаётся энкодер: события и анкета → [USR] [${arch.dim}] — вектор клиента.`,
    beats: ['без головы', 'вектор клиента'],
    regions: {
      embedding: 'context',
      eventEncoder: 'context',
      calendar: 'context',
      profile: 'context',
      history: 'context',
      historyEncoder: 'context',
      clientEmbedding: 'focus',
    },
    shot: ['final', 'finalClose'],
    part: 'clientEmbedding',
  },
  {
    id: 'downstream',
    title: 'Задачи',
    caption: `Вектор клиента на момент T → логистическая регрессия → churn_active90. T train — конец истории обучения, у val — конец выгрузки − ${demo.downstream.horizon_days} дней.`,
    beats: ['вектор на T', 'задачи'],
    regions: { downstream: 'focus' },
    shot: 'downstream',
    part: 'downstream',
  },
  {
    id: 'wave4',
    title: 'Волна 4',
    caption: 'Эксперименты: что уже в эталоне, что запущено и что ждёт. Статусы — из каталогов прогонов, решения — только вручную.',
    beats: ['эксперименты'],
    regions: {},
    screen: 'wave4',
  },
  {
    id: 'insight',
    title: 'Главный вывод',
    caption: 'Низкий MLM loss ещё не значит полезный вектор клиента. Поэтому E1 даёт [USR] свою цель — это гипотеза, её нужно проверить.',
    beats: ['волна 1', 'MLM ≠ представление', 'E1'],
    regions: {},
    screen: 'insight',
  },
]

// Маска и её механизмы — для подписей MLM.
export const MASK_TEXT = `event ${Math.round(Number(masking.event_probability) * 100)}% · key ${Math.round(Number(masking.key_probability) * 100)}% · value ≈${Math.round(Number(masking.value_probability) * 100)}% (${num(Number(masking.min_value_probability), 2)}–${num(Number(masking.max_value_probability), 2)})`

export function shotOf(step: Step, beat: number): ShotId | undefined {
  if (!step.shot) return undefined
  if (Array.isArray(step.shot)) return step.shot[Math.min(beat, step.shot.length - 1)]
  return step.shot
}
