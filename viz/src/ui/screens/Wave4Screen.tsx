import { useEffect, useState } from 'react'
import decisions from '../../data/wave4_decisions.json'
import { ACCEPTED_BASELINE, EXPERIMENTS, MEASUREMENT_FIXES, PROTOCOL, SOURCE } from '../../content/wave'

// ============================================================
// ШАГ 17. ВОЛНА 4: ЭКСПЕРИМЕНТЫ И СТАТУСЫ
// ============================================================
//
// Статус — из каталога прогона data/runs/w4-<id> (сервер /api/wave4):
// NOT RUN, RUNNING, DONE или INTERRUPTED (записей давно нет, эпохи
// не все). ACCEPTED и REJECTED — только из wave4_decisions.json.
// Результатов здесь не придумывается: показан лучший val loss,
// если прогон его записал, — вторичный критерий протокола.
// ============================================================

interface RunStatus {
  id: string
  status: 'NOT_RUN' | 'RUNNING' | 'DONE' | 'STALLED'
  epoch: number | null
  epochs: number | null
  step: number
  totalSteps: number | null
  bestValLoss: number | null
  minutesSinceWrite: number | null
  checkpoint: boolean
  stopReason: string | null
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

export default function Wave4Screen() {
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
    <div className="screen" onClick={(event) => event.stopPropagation()}>
      <h1>Волна 4</h1>
      <p className="sub">Эксперименты программы аудита. Статус — по каталогу прогона data/runs/w4-&lt;id&gt;; решение — только вручную.</p>
      {error ? <div className="banner">{error}</div> : null}

      <div className="two">
        <div className="card">
          <table className="grid">
            <thead>
              <tr>
                <th>#</th>
                <th>эксперимент</th>
                <th>как включить</th>
                <th>пересобрать</th>
                <th>статус</th>
                <th>прогон</th>
              </tr>
            </thead>
            <tbody>
              {EXPERIMENTS.map((item) => {
                const run = statuses[item.id]
                const decision = DECISIONS[item.id]?.decision
                const status = decision ?? run?.status ?? 'NOT_RUN'
                return (
                  <tr key={item.id}>
                    <td className="mono">{item.id}</td>
                    <td>
                      {item.title}
                      <div style={{ color: 'var(--muted)', fontSize: 12, marginTop: 2 }}>гипотеза: {item.hypothesis}</div>
                    </td>
                    <td className="mono" style={{ fontSize: 12, color: 'var(--text-2)' }}>
                      {item.enable}
                    </td>
                    <td className="mono" style={{ fontSize: 12, color: 'var(--text-2)' }}>
                      {item.rebuild}
                    </td>
                    <td>
                      <span className={`chip ${status}`}>{LABEL[status]}</span>
                    </td>
                    <td className="mono" style={{ fontSize: 12, color: 'var(--text-2)' }}>
                      {run && run.status !== 'NOT_RUN' ? (
                        <>
                          эпоха {run.epoch ?? '—'}/{run.epochs ?? '—'}
                          {run.bestValLoss !== null ? ` · val ${run.bestValLoss.toFixed(4)}` : ''}
                          {run.status === 'STALLED' ? ` · нет записей ${run.minutesSinceWrite} мин` : ''}
                        </>
                      ) : (
                        '—'
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <p className="source">
            Источник: {SOURCE}. val loss — вторичный критерий: у E3 и E6 и между разными масками он несравним.
          </p>
        </div>

        <div>
          <div className="card">
            <h3>ТЕКУЩИЙ ПРИНЯТЫЙ ЭТАЛОН</h3>
            <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.6 }}>
              {ACCEPTED_BASELINE.map((item) => (
                <li key={item.id}>
                  <span className="chip ACCEPTED" style={{ marginRight: 8 }}>
                    {item.id}
                  </span>
                  {item.text}
                </li>
              ))}
            </ul>
            <p className="source">Приняты решением владельца без A/B.</p>
          </div>
          <div className="card" style={{ marginTop: 14 }}>
            <h3>ИСПРАВЛЕНИЯ ИЗМЕРЕНИЯ В ЭТАЛОНЕ</h3>
            <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.6, color: 'var(--text-2)' }}>
              {MEASUREMENT_FIXES.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          <div className="card" style={{ marginTop: 14 }}>
            <h3>ПРОТОКОЛ РЕШЕНИЯ</h3>
            <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.6, color: 'var(--text-2)' }}>
              {PROTOCOL.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
