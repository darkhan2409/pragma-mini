import { motion } from 'motion/react'
import { demo, type Interval } from '../../data/demo'
import { useStore } from '../../store'

// ============================================================
// ШАГ 18. ГЛАВНЫЙ ВЫВОД
// ============================================================
//
// Бит 0 — три сценария прогона на val из экспорта: handcrafted →
// CatBoost, [USR] → регрессия, handcrafted и [USR] → CatBoost, с
// парной разницей к первому и 95% интервалом. Вывод — по интервалу,
// не руками. Бит 1 — MLM quality ≠ representation quality; val loss
// B0 из экспорта прогона. Бит 2 — E1 как гипотеза, без обещаний.
// ============================================================

const LABEL: Record<string, string> = {
  catboost: 'handcrafted → CatBoost',
  usr: '[USR] → LogisticRegression',
  catboost_plus_usr: 'handcrafted + [USR] → CatBoost',
}

function signed(value: number): string {
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`
}

function interval(delta: Interval): string {
  return `${signed(delta.mean)} [${signed(delta.low)}, ${signed(delta.high)}]`
}

// Что говорит интервал разницы с CatBoost на handcrafted-признаках.
function verdict(delta: Interval): string {
  if (delta.low > 0) return '[USR] даёт сигнал сверх handcrafted-признаков'
  if (delta.high < 0) return 'с [USR] модель хуже'
  return '[USR] почти ничего не добавляет'
}

function Scenarios() {
  const scenarios = demo.downstream.scenarios

  if (!scenarios) {
    return (
      <div className="card">
        <h3>ТРИ СЦЕНАРИЯ</h3>
        <p className="source">Сравнения в экспорте нет: python -m src.downstream.probe --tag &lt;прогон&gt;, затем export_demo.py --run.</p>
      </div>
    )
  }

  return (
    <div className="card">
      {Object.entries(scenarios.tasks).map(([task, block]) => {
        const plus = block.cells.catboost_plus_usr?.vs_reference
        return (
          <div key={task}>
            <h3>{`ТРИ СЦЕНАРИЯ — ${task}, ${scenarios.group.toUpperCase()}`}</h3>
            <table className="grid">
              <thead>
                <tr>
                  <th>сценарий</th>
                  <th style={{ textAlign: 'right' }}>PR-AUC</th>
                  <th style={{ textAlign: 'right' }}>ROC-AUC</th>
                  <th style={{ textAlign: 'right' }}>log-loss</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(block.cells).map(([name, cell]) => (
                  <tr key={name} style={name === 'usr' ? { background: 'rgba(240,180,76,0.07)' } : undefined}>
                    <td className="mono">{LABEL[name] ?? name}</td>
                    <td className="num">{cell.pr_auc.toFixed(3)}</td>
                    <td className="num">{cell.roc_auc.toFixed(3)}</td>
                    <td className="num">{cell.log_loss.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {Object.entries(block.cells).map(([name, cell]) =>
              cell.vs_reference ? (
                <p key={name} style={{ marginTop: 8 }}>
                  {`${LABEL[name] ?? name} − ${LABEL[block.reference] ?? block.reference}: PR-AUC ${interval(cell.vs_reference.pr_auc)}`}
                  {name === 'catboost_plus_usr' && plus ? ` — ${verdict(plus.pr_auc)}.` : '.'}
                </p>
              ) : null,
            )}
            <p className="source">
              {`${block.rows} клиентов, ${block.positives} ушедших; парный bootstrap, 95% интервал. Источник: ${scenarios.source}.`}
            </p>
          </div>
        )
      })}
    </div>
  )
}

export default function InsightScreen() {
  const beat = useStore((s) => s.beat)
  const run = demo.run

  return (
    <div className="screen" onClick={(event) => event.stopPropagation()}>
      <h1>Low MLM loss ≠ useful Client Embedding</h1>
      <p className="sub">Хорошее предсказание скрытых токенов ещё не делает вектор клиента полезным для задач.</p>

      <div className="two">
        <Scenarios />

        <div>
          {beat >= 1 ? (
            <motion.div className="card" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
              <div className="neq">
                <span className="a">MLM quality</span>
                <span className="op">≠</span>
                <span className="b">representation</span>
              </div>
              {run ? (
                <p style={{ color: 'var(--text-2)' }}>
                  {run.run}: val loss по эпохам{' '}
                  <span className="mono" style={{ color: 'var(--text)' }}>
                    {run.epochs.map((item) => item.val_loss?.toFixed(3)).join(' → ')}
                  </span>
                  . Это качество MLM, а не вектора клиента: на задаче его [USR] — слева.
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
