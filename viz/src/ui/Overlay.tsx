import { AnimatePresence, motion } from 'motion/react'
import { demo } from '../data/demo'
import { PARTS } from '../content/parts'
import { STEPS, type Illustration } from '../content/steps'
import { useStore } from '../store'

// ============================================================
// ИНТЕРФЕЙС ПОВЕРХ СЦЕНЫ
// ============================================================

const ILLUSTRATION: Record<Illustration, string> = {
  vectors: 'значения векторов',
  attention: 'веса внимания',
  qk: 'направления Q и K',
  topk: 'вероятности предсказаний',
  gradients: 'толщина градиентов',
}

function Topbar() {
  const mode = useStore((s) => s.mode)
  const setMode = useStore((s) => s.setMode)
  const togglePanel = useStore((s) => s.togglePanel)
  const step = useStore((s) => STEPS[s.step])
  const index = useStore((s) => s.step)

  const illustrative = mode === 'present' && step.illustrative?.length ? step.illustrative : null
  const real = mode === 'present' && demo.model && step.checkpoint?.length ? step.checkpoint : null

  return (
    <div className="topbar">
      <div className="brand">
        <b>PRAGMA</b> · pipeline
      </div>
      <div className="modes" onClick={(event) => event.stopPropagation()}>
        <button className={mode === 'present' ? 'active' : ''} onClick={() => setMode('present')}>
          PRESENT
        </button>
        <button className={mode === 'explore' ? 'active' : ''} onClick={() => setMode('explore')}>
          EXPLORE
        </button>
      </div>
      <div className="spacer" />
      {mode === 'present' && step.screen ? (
        <div className="stepcount">
          {String(index + 1).padStart(2, '0')} / {STEPS.length} · {step.title}
        </div>
      ) : null}
      {real && demo.model ? (
        <div className="badge real">
          ИЗ ЧЕКПОЙНТА {demo.model.checkpoint.split('/').slice(-2, -1)[0]}, эпоха {demo.model.epoch}: {real.map((item) => ILLUSTRATION[item]).join(', ')}
        </div>
      ) : null}
      {illustrative ? <div className="badge">ИЛЛЮСТРАЦИЯ — не из модели: {illustrative.map((item) => ILLUSTRATION[item]).join(', ')}</div> : null}
      <div className="modes" onClick={(event) => event.stopPropagation()}>
        <button onClick={() => togglePanel('steps')}>ШАГИ · S</button>
        <button onClick={() => togglePanel('help')}>?</button>
      </div>
    </div>
  )
}

function Caption() {
  const index = useStore((s) => s.step)
  const beat = useStore((s) => s.beat)
  const captions = useStore((s) => s.captions)
  const step = STEPS[index]

  if (!captions || step.screen) return null

  return (
    <div className="caption">
      <AnimatePresence mode="wait">
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
        >
          <div className="counter">
            {String(index + 1).padStart(2, '0')} / {STEPS.length} · {step.beats[beat]}
          </div>
          <h1>{step.title}</h1>
          <p>{step.caption}</p>
        </motion.div>
      </AnimatePresence>
      <div className="dots">
        {STEPS.map((item, i) => (
          <span key={item.id} className={i === index ? 'now' : i < index ? 'done' : ''} />
        ))}
      </div>
    </div>
  )
}

function Formula() {
  const step = useStore((s) => STEPS[s.step])
  const beat = useStore((s) => s.beat)
  const mode = useStore((s) => s.mode)
  const show = mode === 'present' && step.id === 'embedding' && beat >= 2
  const d = demo.architecture.dim

  return (
    <AnimatePresence>
      {show ? (
        <motion.div className="formula" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.4 }}>
          <div>
            x = √d·E[key] + √d·E[value] + P(piece)
          </div>
          <div className="dim">
            d = {d}, √d ≈ {Math.sqrt(d).toFixed(2)} · P — фиксированная синусоида номера куска внутри значения
          </div>
          <div className="dim">[EVT] и [USR]: только √d·E[маркер] · календарь и время сюда не входят</div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  )
}

function StepList() {
  const go = useStore((s) => s.go)
  const current = useStore((s) => s.step)
  const setMode = useStore((s) => s.setMode)
  const togglePanel = useStore((s) => s.togglePanel)

  return (
    <div className="drawer left steplist" onClick={(event) => event.stopPropagation()}>
      <button className="close" onClick={() => togglePanel('steps')}>
        ×
      </button>
      <h2>Шаги</h2>
      {STEPS.map((step, index) => (
        <button
          key={step.id}
          className={index === current ? 'now' : ''}
          onClick={() => {
            setMode('present')
            go(index)
            togglePanel('steps')
          }}
        >
          <span className="n">{index + 1}</span>
          {step.title}
        </button>
      ))}
    </div>
  )
}

function Help() {
  const togglePanel = useStore((s) => s.togglePanel)
  const rows: [string, string][] = [
    ['→ · Space · PageDown · клик', 'дальше'],
    ['← · PageUp · Backspace', 'назад'],
    ['Home · End', 'первый · последний шаг'],
    ['S', 'список шагов'],
    ['E · P', 'EXPLORE · PRESENT'],
    ['H', 'скрыть подписи'],
    ['F', 'полный экран'],
    ['L', 'режим низкой мощности'],
    ['EXPLORE: мышь', 'вращение, сдвиг правой, зум колесом'],
    ['EXPLORE: клик по блоку', 'детали; Esc — закрыть'],
    ['R', 'общий вид (EXPLORE)'],
  ]

  return (
    <div className="drawer right" onClick={(event) => event.stopPropagation()}>
      <button className="close" onClick={() => togglePanel('help')}>
        ×
      </button>
      <h2>Управление</h2>
      <div className="help-grid" style={{ marginTop: 14 }}>
        {rows.map(([key, text]) => (
          <div key={key} style={{ display: 'contents' }}>
            <kbd>{key}</kbd>
            <span style={{ color: 'var(--text-2)' }}>{text}</span>
          </div>
        ))}
      </div>
      <h3>ИСТОЧНИКИ ЧИСЕЛ</h3>
      <p>
        Архитектура, словарь, клиент и план обучения — экспорт <code>viz/scripts/export_demo.py</code> от {demo.sources.generated_at.slice(0, 10)}.
        Помеченное «иллюстрация» — схема, не выход модели.
        {demo.model ? ` Помеченное «из чекпойнта» посчитано обученной моделью ${demo.model.checkpoint}, эпоха ${demo.model.epoch}.` : ''}
      </p>
    </div>
  )
}

function DetailPanel() {
  const selected = useStore((s) => s.selected)
  const select = useStore((s) => s.select)
  const setMode = useStore((s) => s.setMode)
  const go = useStore((s) => s.go)

  const part = selected ? PARTS[selected] : undefined

  return (
    <AnimatePresence>
      {part ? (
        <motion.div
          key={selected}
          className="drawer right"
          initial={{ opacity: 0, x: 24 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 24 }}
          transition={{ duration: 0.3 }}
          onClick={(event) => event.stopPropagation()}
        >
          <button className="close" onClick={() => select(null)}>
            ×
          </button>
          <h2>{part.title}</h2>
          <div className="code">{part.code}</div>
          <h3>ФАКТЫ</h3>
          <table>
            <tbody>
              {part.facts.map(([key, value]) => (
                <tr key={key}>
                  <td>{key}</td>
                  <td>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>КАК УСТРОЕНО</h3>
          <ul>
            {part.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
          <button
            className="linkbtn"
            onClick={() => {
              const index = STEPS.findIndex((step) => step.id === part.step)
              setMode('present')
              go(Math.max(0, index))
            }}
          >
            Показать в презентации →
          </button>
        </motion.div>
      ) : null}
    </AnimatePresence>
  )
}

export function Overlay() {
  const mode = useStore((s) => s.mode)
  const panel = useStore((s) => s.panel)
  const screen = useStore((s) => (s.mode === 'present' ? STEPS[s.step].screen : undefined))

  return (
    <div className="overlay">
      <Topbar />
      {mode === 'present' ? <Caption /> : null}
      <Formula />
      {mode === 'explore' ? (
        <div className="hint">EXPLORE · мышь — вращение и зум · клик по блоку — детали · R — общий вид · P — презентация</div>
      ) : null}
      {mode === 'explore' ? <DetailPanel /> : null}
      {panel === 'steps' ? <StepList /> : null}
      {panel === 'help' ? <Help /> : null}
      {screen ? null : (
        <div className="footer">
          экспорт {demo.sources.generated_at.slice(0, 16).replace('T', ' ')} · словарь {demo.sources.vocabulary_digest.slice(0, 8)} · клиент{' '}
          {demo.client.client_id}
        </div>
      )}
    </div>
  )
}
