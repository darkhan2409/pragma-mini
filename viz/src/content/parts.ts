import { demo, int } from '../data/demo'

// ============================================================
// ДЕТАЛИ ДЛЯ EXPLORE
// ============================================================
//
// Панель, которая открывается по клику на блок: имя компонента
// в коде, где он лежит, формы и параметры. Числа — из экспорта,
// текст — короткие факты, сверенные с кодом.
// ============================================================

export interface Part {
  title: string
  code: string
  facts: [string, string][]
  notes: string[]
  step: string
}

const arch = demo.architecture
const enc = arch.encoders
const t = demo.training
const m = demo.masking

function params(n: number): string {
  return int(n)
}

export const PARTS: Record<string, Part> = {
  client: {
    title: 'Клиент и его лента',
    code: 'src/preprocessing/read.py · Group.history',
    facts: [
      ['клиент', demo.client.client_id],
      ['событий', int(demo.client.n_events)],
      ['токенов событий', int(demo.client.n_tokens)],
      ['токенов анкеты', int(demo.client.n_profile_tokens)],
      ['cutoff T', demo.dataset.cutoff.slice(0, 10)],
    ],
    notes: ['События строго раньше cutoff; анкета — состояние на тот же cutoff.', 'Цвет точки — источник выгрузки; модели он не подаётся.'],
    step: 'client',
  },
  event: {
    title: 'Событие',
    code: 'src/preprocessing/keys.py · смысловые ключи',
    facts: [
      ['тип', demo.client.highlighted.hero.type],
      ['источник', demo.client.highlighted.hero.source],
      ['время (банк)', demo.client.highlighted.hero.local_time],
    ],
    notes: ['Поля источника переводятся в смысловые ключи.', 'Время — отдельный канал модели, не поле события.'],
    step: 'event',
  },
  tokens: {
    title: 'Токенизация',
    code: 'src/tokenization/encode.py · encode_event',
    facts: [
      ['словарь', int(demo.vocabulary.size)],
      ['ключей', int(demo.vocabulary.kinds.key.count)],
      ['категорий', int(demo.vocabulary.kinds.value.count)],
      ['диапазонов', int(demo.vocabulary.kinds.bucket.count)],
      ['кусков BPE', int(demo.vocabulary.kinds.bpe.count)],
    ],
    notes: [
      '[EVT] первым, затем пары в порядке key_id.',
      'Позиция 0 — начало значения; куски BPE одного значения идут 0, 1, 2…',
      'Число — номер диапазона (со split_by — среди диапазонов своего условия).',
    ],
    step: 'tokens',
  },
  embedding: {
    title: 'InputEmbedding',
    code: 'src/embedding/layer.py',
    facts: [
      ['таблица', `${int(arch.vocab_size)} × ${arch.dim}`],
      ['параметров', params(arch.embedding_parameters)],
      ['множитель', `√${arch.dim} ≈ ${Math.sqrt(arch.dim).toFixed(2)}`],
    ],
    notes: [
      'x = √d·E[key] + √d·E[value] + P(piece).',
      'P — фиксированная синусоида номера куска внутри значения, не обучается.',
      '[EVT] и [USR] — только √d·E[маркер], без второго слагаемого и позиции.',
      'Одна таблица на ключи, значения, диапазоны, куски и служебные токены.',
    ],
    step: 'embedding',
  },
  eventEncoder: {
    title: 'EventEncoder',
    code: 'src/event/encoder.py',
    facts: [
      ['блоков', String(enc.event.blocks)],
      ['голов', String(enc.event.config.heads)],
      ['FFN', String(enc.event.config.feedforward)],
      ['dropout', String(enc.event.config.dropout)],
      ['параметров', params(enc.event.parameters)],
    ],
    notes: [
      'nn.TransformerEncoderLayer: pre-norm, GELU; финальный LayerNorm.',
      'Внимание только внутри события (cu_seqlens), позиций и времени нет.',
      'Выход — столбец [EVT]; векторы токенов уходят в MLM-голову.',
    ],
    step: 'eventEncoder',
  },
  calendar: {
    title: 'Календарь',
    code: 'src/event/encoder.py · EventEncoder.calendar',
    facts: [
      ['признаки', demo.dataset.calendar.features.length.toString()],
      ['циклы', Object.values(demo.dataset.calendar.cycles).join(' / ')],
      ['параметров', params(enc.event.parts.calendar ?? 0)],
    ],
    notes: ['Linear 6→128 · GELU · Linear 128→128.', 'dated = event + calendar — после финального LN энкодера.', 'Местное время банка; неделя с понедельника.'],
    step: 'calendar',
  },
  profile: {
    title: 'ProfileEncoder',
    code: 'src/profile/encoder.py',
    facts: [
      ['блоков', String(enc.profile.blocks)],
      ['голов', String(enc.profile.config.heads)],
      ['полей анкеты', String(demo.dataset.profile_fields.length)],
      ['типов вех', String(demo.dataset.lifelong_types.length)],
      ['параметров', params(enc.profile.parameters)],
    ],
    notes: [
      '[USR], поля анкеты по key_id, затем вехи по времени.',
      'TimeRoPE по давности вех; у [USR] и полей позиция 0.',
      'Выход — столбец [USR]: вектор анкеты.',
    ],
    step: 'profile',
  },
  history: {
    title: 'Последовательность истории',
    code: 'src/history/encoder.py · src/temporal/position.py',
    facts: [
      ['позиция [USR]', '0'],
      ['позиция события', '8·log1p(Δt/8), Δt в секундах до T'],
      ['точка отсчёта', demo.dataset.time_anchor],
    ],
    notes: ['Вектор анкеты кладётся в слот [USR] без повторного эмбеддинга.', 'События — от старых к новым; каждое уже с календарём.'],
    step: 'history',
  },
  rope: {
    title: 'TimeRoPE',
    code: 'src/history/encoder.py · TimeRoPE',
    facts: [
      ['база', '10000'],
      ['head_dim', '32'],
      ['частоты', '16'],
    ],
    notes: ['θᵢ = pos · 10000^(−2i/32).', 'Поворачиваются только Q и K, V — нет.', 'q·k зависит от разности позиций.'],
    step: 'rope',
  },
  historyEncoder: {
    title: 'HistoryEncoder',
    code: 'src/history/encoder.py',
    facts: [
      ['блоков', String(enc.history.blocks)],
      ['голов × head_dim', `${enc.history.config.heads} × ${arch.dim / Number(enc.history.config.heads)}`],
      ['FFN', String(enc.history.config.feedforward)],
      ['параметров', params(enc.history.parameters)],
    ],
    notes: ['Внимание двунаправленное.', '[USR] после финального LN — Client Embedding.', 'Слоты событий — векторы событий с учётом истории.'],
    step: 'historyEncoder',
  },
  mlm: {
    title: 'MlmHead',
    code: 'src/mlm/model.py',
    facts: [
      ['вход', `3 × ${arch.dim} = ${3 * arch.dim}`],
      ['Linear', `${3 * arch.dim} → ${arch.dim}`],
      ['параметров', params(arch.head_parameters)],
      ['label smoothing', String(t.label_smoothing)],
    ],
    notes: [
      'Входы: вектор токена после Event Encoder, вектор его события и [USR] после истории.',
      'Логиты = h·Eᵀ той же таблицей эмбеддингов, без √d.',
      `Маска: event ${m.event_probability}, key ${m.key_probability}, value ${m.value_probability} в среднем; 10% выбранных — [UNK] без метки.`,
    ],
    step: 'mlm',
  },
  backprop: {
    title: 'Обучение',
    code: 'src/mlm/train.py',
    facts: [
      ['оптимизатор', 'AdamW'],
      ['learning rate', String(t.learning_rate)],
      ['weight decay', String(t.weight_decay)],
      ['clip', String(t.max_grad_norm)],
    ],
    notes: ['Градиент — среднее по всем целям окна.', 'Норма в телеметрии — до обрезки.'],
    step: 'backprop',
  },
  loop: {
    title: 'Цикл обучения',
    code: 'src/mlm/train.py · src/mlm/inputs.py',
    facts: [
      ['token_budget', int(Number(t.token_budget))],
      ['grad_accum_steps', String(t.grad_accum_steps)],
      ['micro-batch в эпохе', int(demo.batching.micro_batches)],
      ['токенов в среднем', int(demo.batching.mean_tokens)],
      ['warmup', String(t.warmup_steps)],
      ['LR', `${t.learning_rate} → ${t.min_learning_rate}`],
    ],
    notes: ['Цена клиента — токены событий и анкеты плюс [USR].', 'Клиент дороже бюджета идёт отдельным micro-batch.', 'bf16 и varlen FlashAttention на CUDA.'],
    step: 'loop',
  },
  clientEmbedding: {
    title: 'Client Embedding',
    code: 'src/mlm/model.py · Model.readouts',
    facts: [
      ['размерность', String(arch.dim)],
      ['всего параметров', params(arch.total_parameters)],
    ],
    notes: ['usr — [USR] после истории; ещё profile, mean_event и last_event.', 'MLM-голова в вектор клиента не входит.'],
    step: 'final',
  },
  downstream: {
    title: 'Оценка на задачах',
    code: 'src/downstream/probe.py · tasks.py',
    facts: [
      ['задачи', demo.downstream.tasks.join(', ')],
      ['горизонт', `${demo.downstream.horizon_days} дней`],
      ['T val', demo.downstream.cutoffs.val.slice(0, 10)],
    ],
    notes: [
      'Проба — StandardScaler + LogisticRegressionCV (L2).',
      'CatBoost — отдельный бейзлайн churn на агрегатах, не на эмбеддинге.',
      'Фрод на уровне операции не оценивается: вектор нужен строго до каждой операции, а история двунаправленная.',
    ],
    step: 'downstream',
  },
}
