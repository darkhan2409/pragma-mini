import { motion } from 'motion/react'
import { demo, type Interval } from '../../data/demo'
import { useStore } from '../../store'

// ============================================================
// ШАГ 18. ГЛАВНЫЙ ВЫВОД
// ============================================================
//
// Бит 0 — диагностика [USR] прогона на val из экспорта: векторы
// модели и начальных весов × регрессия и CatBoost только на [USR],
// парные разницы с 95% интервалом. Вывод — по интервалу, не руками.
// Бит 1 — MLM quality ≠ representation quality; val loss B0 из
// экспорта прогона. Бит 2 — E1 как гипотеза, без обещаний.
// ============================================================

const HIGHLIGHT = new Set(['usr', 'init:usr'])

function signed(value: number): string {
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`
}

function interval(delta: Interval): string {
  return `${signed(delta.mean)} [${signed(delta.low)}, ${signed(delta.high)}]`
}

// Что говорит интервал разницы: выше нуля, ниже нуля или неотличимо.
function verdict(delta: Interval, better: string, worse: string, same: string): string {
  if (delta.low > 0) return better
  if (delta.high < 0) return worse
  return same
}

function Diagnostic() {
  const diagnostic = demo.downstream.usr_diagnostic

  if (!diagnostic) {
    return (
      <div className="card">
        <h3>ДИАГНОСТИКА [USR]</h3>
        <p className="source">
          Диагностики в экспорте нет: python -m src.downstream.probe --tag &lt;прогон&gt; --control init, затем
          export_demo.py --run.
        </p>
      </div>
    )
  }

  const { tag, control } = diagnostic
  const label: Record<string, string> = {
    [`${control}:usr`]: `${control} [USR] + регрессия`,
    usr: `${tag} [USR] + регрессия`,
    catboost_usr: `${tag} [USR] + CatBoost`,
    [`${control}:catboost_usr`]: `${control} [USR] + CatBoost`,
  }

  return (
    <div className="card">
      {Object.entries(diagnostic.tasks).map(([task, block]) => {
        const pretrain = block.deltas.usr_vs_control
        const head = block.deltas.catboost_vs_lr
        return (
          <div key={task}>
            <h3>{`ДИАГНОСТИКА [USR] — ${task}, ${diagnostic.group.toUpperCase()}`}</h3>
            <table className="grid">
              <thead>
                <tr>
                  <th>модель</th>
                  <th style={{ textAlign: 'right' }}>PR-AUC</th>
                  <th style={{ textAlign: 'right' }}>ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(label)
                  .filter((name) => name in block.cells)
                  .map((name) => (
                    <tr key={name} style={HIGHLIGHT.has(name) ? { background: 'rgba(240,180,76,0.07)' } : undefined}>
                      <td className="mono">{label[name]}</td>
                      <td className="num">{block.cells[name].pr_auc.toFixed(3)}</td>
                      <td className="num">{block.cells[name].roc_auc.toFixed(3)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
            {pretrain ? (
              <p style={{ marginTop: 12 }}>
                {`${tag} − ${control}, регрессия: PR-AUC ${interval(pretrain.pr_auc)} — `}
                {verdict(
                  pretrain.pr_auc,
                  'обучение сделало [USR] полезнее',
                  'обученный [USR] хуже начальных весов',
                  'обученный [USR] не отличим от начальных весов',
                )}
                .
              </p>
            ) : null}
            {head ? (
              <p>
                {`CatBoost − регрессия на [USR] ${tag}: PR-AUC ${interval(head.pr_auc)} — `}
                {verdict(
                  head.pr_auc,
                  'нелинейная голова находит больше',
                  'нелинейная голова находит меньше',
                  'нелинейная голова больше не находит',
                )}
                .
              </p>
            ) : null}
            <p className="source">
              {`${block.rows} клиентов, ${block.positives} ушедших; парный bootstrap, 95% интервал. Источник: ${diagnostic.source}.`}
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
        <Diagnostic />

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
                  . Это качество MLM, а не вектора клиента: на задаче его [USR] сравнивается с [USR] начальных весов слева.
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
