import { motion } from 'motion/react'
import { useEffect, useState } from 'react'
import decisions from '../../data/wave4_decisions.json'
import { CAUSES, EXPERIMENTS, type Check } from '../../content/wave'
import { useStore } from '../../store'

// ============================================================
// ШАГ 20. ПОЧЕМУ ТАК: ВОЗМОЖНЫЕ ПРИЧИНЫ И ПРОВЕРКИ
// ============================================================
//
// Причины открываются по битам. У каждой — что проверить:
// эксперимент волны 4 со статусом его прогона или работа вне
// волны 4. Статус — из каталога прогона data/runs/w4-<id> (сервер
// /api/wave4): NOT RUN, RUNNING, DONE или INTERRUPTED (записей давно
// нет, эпохи не все). ACCEPTED и REJECTED — только из
// wave4_decisions.json. Результатов здесь не придумывается.
// ============================================================

interface RunStatus {
  id: string
  status: 'NOT_RUN' | 'RUNNING' | 'DONE' | 'STALLED'
  epoch: number | null
  epochs: number | null
  bestValLoss: number | null
  minutesSinceWrite: number | null
}

type Decision = { decision: 'ACCEPTED' | 'REJECTED'; date?: string; note?: string }

const DECISIONS = decisions as Record<string, Decision>

const LABEL: Record<string, string> = {
  NOT_RUN: 'NOT RUN',
  RUNNING: 'RUNNING',
  DONE: 'DONE',
  STALLED: 'INTERRUPTED',
  ACCEPTED: 'ACCEPTED',
  REJECTED: 'REJECTED',
}

const TITLE = Object.fromEntries(EXPERIMENTS.map((item) => [item.id, item]))

function Status({ id, statuses }: { id: string; statuses: Record<string, RunStatus> }) {
  const run = statuses[id]
  const status = DECISIONS[id]?.decision ?? run?.status ?? 'NOT_RUN'
  return (
    <>
      <span className={`chip ${status}`}>{LABEL[status]}</span>
      {run && run.status !== 'NOT_RUN' ? (
        <span className="mono" style={{ fontSize: 12, color: 'var(--text-2)', marginLeft: 8 }}>
          эпоха {run.epoch ?? '—'}/{run.epochs ?? '—'}
          {run.bestValLoss !== null ? ` · val ${run.bestValLoss.toFixed(4)}` : ''}
          {run.status === 'STALLED' ? ` · нет записей ${run.minutesSinceWrite} мин` : ''}
        </span>
      ) : null}
    </>
  )
}

function CheckLine({ check, statuses }: { check: Check; statuses: Record<string, RunStatus> }) {
  const experiment = check.id ? TITLE[check.id] : null
  return (
    <li style={{ marginBottom: 6 }}>
      {experiment ? (
        <>
          <span className="mono" style={{ color: 'var(--text)' }}>
            {experiment.id}
          </span>{' '}
          {experiment.title}
          <span className="mono" style={{ fontSize: 12, color: 'var(--muted)' }}>
            {' '}
            · {experiment.enable}
          </span>
          <div style={{ color: 'var(--text-2)', fontSize: 13 }}>{check.text}</div>
          <div style={{ marginTop: 4 }}>
            <Status id={experiment.id} statuses={statuses} />
          </div>
        </>
      ) : (
        <>
          {check.text}
          <div style={{ marginTop: 4 }}>
            <span className="chip">ВНЕ ВОЛНЫ 4</span>
          </div>
        </>
      )}
    </li>
  )
}

export default function CausesScreen() {
  const beat = useStore((s) => s.beat)
  const [statuses, setStatuses] = useState<Record<string, RunStatus>>({})
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const ids = EXPERIMENTS.map((item) => item.id).join(',')
    let alive = true
    const load = () =>
      fetch(`/api/wave4?ids=${ids}`)
        .then((response) => response.json())
        .then((body: { statuses: RunStatus[] }) => {
          if (alive) setStatuses(Object.fromEntries(body.statuses.map((item) => [item.id, item])))
        })
        .catch((reason) => alive && setError(`API недоступно: ${String(reason)} — статусы не показаны`))
    void load()
    const timer = setInterval(load, 10000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])

  return (
    <div className="screen">
      <h1>Почему так получилось</h1>
      <p className="sub">Возможные причины, почему [USR] после MLM слабее handcrafted-признаков. Это гипотезы, не выводы; у каждой — что проверить.</p>
      {error ? <div className="banner">{error}</div> : null}

      <div style={{ maxWidth: 980 }}>
        {CAUSES.map((cause, index) =>
          index <= beat ? (
            <motion.div
              key={cause.title}
              className="card"
              style={{ marginBottom: 14 }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
            >
              <h2 className="cause-title">
                <span className="num">{index + 1}</span>
                {cause.title}
              </h2>
              <p style={{ marginTop: 0 }}>{cause.why}</p>
              <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 6 }}>ПРОВЕРИТЬ</div>
              <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.5 }}>
                {cause.checks.map((check) => (
                  <CheckLine key={check.id ?? check.text} check={check} statuses={statuses} />
                ))}
              </ul>
            </motion.div>
          ) : null,
        )}
      </div>
    </div>
  )
}
