import { motion } from 'motion/react'
import { demo } from '../../data/demo'
import { SOURCE, WAVE1 } from '../../content/wave'
import { useStore } from '../../store'

// ============================================================
// ШАГ 18. ГЛАВНЫЙ ВЫВОД
// ============================================================
//
// Бит 0 — таблица волны 1 (дословно из аудита, с оговоркой).
// Бит 1 — MLM quality ≠ representation quality; val loss B0 из
// экспорта прогона. Бит 2 — E1 как гипотеза, без обещаний.
// ============================================================

const HIGHLIGHT = new Set(['usr', 'init:usr'])

export default function InsightScreen() {
  const beat = useStore((s) => s.beat)
  const run = demo.run

  return (
    <div className="screen" onClick={(event) => event.stopPropagation()}>
      <h1>Low MLM loss ≠ useful Client Embedding</h1>
      <p className="sub">Хорошее предсказание скрытых токенов ещё не делает вектор клиента полезным для задач.</p>

      <div className="two">
        <div className="card">
          <h3>{WAVE1.caption.toUpperCase()}</h3>
          <table className="grid">
            <thead>
              <tr>
                <th>набор</th>
                {WAVE1.columns.map((column) => (
                  <th key={column} style={{ textAlign: 'right' }}>
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {WAVE1.rows.map((row) => (
                <tr key={row.set} style={HIGHLIGHT.has(row.set) ? { background: 'rgba(240,180,76,0.07)' } : undefined}>
                  <td className="mono">{row.set}</td>
                  {row.values.map((value, index) => (
                    <td key={index} className="num">
                      {value === null ? '—' : value.toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ marginTop: 12 }}>{WAVE1.conclusion}</p>
          <p className="source">Источник: {SOURCE}, раздел «Что уже известно». Данные и код с тех пор менялись.</p>
        </div>

        <div>
          {beat >= 1 ? (
            <motion.div className="card" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
              <div className="neq">
                <span className="a">MLM quality</span>
                <span className="op">≠</span>
                <span className="b">representation</span>
              </div>
              <p style={{ color: 'var(--text-2)' }}>
                В волне 1 обученный [USR] дал PR-AUC не выше, чем необученный, а простые счётчики событий обогнали все векторы.
              </p>
              {run ? (
                <p style={{ color: 'var(--text-2)' }}>
                  Текущий B0 ({run.run}): val loss по эпохам{' '}
                  <span className="mono" style={{ color: 'var(--text)' }}>
                    {run.epochs.map((item) => item.val_loss?.toFixed(3)).join(' → ')}
                  </span>
                  . Это качество MLM; пробы на T для него ещё не посчитаны.
                </p>
              ) : null}
            </motion.div>
          ) : null}

          {beat >= 2 ? (
            <motion.div className="card" style={{ marginTop: 14 }} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
              <h3>E1 — ГИПОТЕЗА, ЕЁ НУЖНО ПРОВЕРИТЬ</h3>
              <p>
                Дать [USR] свою цель: предсказать доли типов событий клиента за последние 7, 30 и 90 дней до точки отсчёта.
              </p>
              <p style={{ color: 'var(--text-2)' }}>
                Голова Linear(128 → типы × 3) на [USR] после истории; потеря — cross-entropy долей, вес usr_aux_weight (по умолчанию 0 — выключено).
                Решение — по протоколу волны 4 на val, не по MLM loss.
              </p>
              <p className="source">Ожидаемый эффект не заявляется: результат появится только после прогона и проб.</p>
            </motion.div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
