import { motion } from 'motion/react'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipContentProps } from 'recharts'
import { demo, type ScenarioTask } from '../../data/demo'
import { useStore } from '../../store'

// ============================================================
// ШАГ 19. ГЛАВНЫЙ ВЫВОД
// ============================================================
//
// Бит 0 — три сценария прогона на val из экспорта: handcrafted →
// CatBoost, [USR] → регрессия, handcrafted и [USR] → CatBoost, и
// короткое пояснение метрик. Внизу — те же PR-AUC и ROC-AUC
// столбиками: сценарий только на [USR] выделен цветом, остальные —
// серые. Бит 1 — MLM quality ≠ representation quality; val loss B0
// из экспорта прогона. Почему так и что проверить — шаг 20.
// ============================================================

const LABEL: Record<string, string> = {
  catboost: 'handcrafted → CatBoost',
  usr: '[USR] → LogisticRegression',
  catboost_plus_usr: 'handcrafted + [USR] → CatBoost',
}

// Выделен сценарий, о котором вывод; остальные — контекст.
const ACCENT = '#BF8601'
const CONTEXT = '#5B6672'
const GRID = '#1E2733'
const TICK = { fill: '#9AA5B1', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }

interface Row {
  key: string
  label: string
  pr_auc: number
  roc_auc: number
}

type Metric = 'pr_auc' | 'roc_auc'

const METRIC: Record<Metric, string> = { pr_auc: 'PR-AUC', roc_auc: 'ROC-AUC' }

function BarTooltip({ active, payload, metric }: TooltipContentProps<number, string> & { metric: Metric }) {
  if (!active || !payload?.length) return null
  const row = payload[0].payload as Row
  return (
    <div style={{ background: '#0D1117', border: '1px solid #1E2733', borderRadius: 8, padding: '8px 10px', fontSize: 12 }}>
      <div style={{ color: '#9AA5B1', marginBottom: 4 }}>{row.label}</div>
      <div>
        <b style={{ color: '#E6EBF1', fontFamily: 'JetBrains Mono, monospace', fontWeight: 500 }}>{row[metric].toFixed(3)}</b>
        <span style={{ color: '#9AA5B1' }}> {METRIC[metric]}</span>
      </div>
    </div>
  )
}

function MetricChart({ rows, metric }: { rows: Row[]; metric: Metric }) {
  return (
    <div>
      <h3>{METRIC[metric]}</h3>
      <ResponsiveContainer width="100%" height={40 * rows.length + 40}>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 52, bottom: 4, left: 4 }}>
          <CartesianGrid stroke={GRID} horizontal={false} />
          <XAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={TICK} stroke={GRID} />
          <YAxis type="category" dataKey="label" width={210} tick={{ ...TICK, fill: '#E6EBF1' }} stroke={GRID} />
          <Tooltip cursor={{ fill: 'rgba(255,255,255,0.03)' }} content={(props) => <BarTooltip {...(props as TooltipContentProps<number, string>)} metric={metric} />} />
          <Bar dataKey={metric} barSize={22} radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {rows.map((row) => (
              <Cell key={row.key} fill={row.key === 'usr' ? ACCENT : CONTEXT} />
            ))}
            <LabelList dataKey={metric} position="right" formatter={(value) => Number(value).toFixed(3)} fill="#E6EBF1" fontSize={12} fontFamily="JetBrains Mono, monospace" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function ScenarioChart({ task, block }: { task: string; block: ScenarioTask }) {
  const rows: Row[] = Object.entries(block.cells).map(([key, cell]) => ({
    key,
    label: LABEL[key] ?? key,
    pr_auc: cell.pr_auc,
    roc_auc: cell.roc_auc,
  }))
  return (
    <div className="card" style={{ marginTop: 14 }}>
      <h3>{`ТРИ СЦЕНАРИЯ НА ГРАФИКЕ — ${task}, VAL · больше — лучше`}</h3>
      <div className="charts" style={{ marginTop: 4 }}>
        <MetricChart rows={rows} metric="pr_auc" />
        <MetricChart rows={rows} metric="roc_auc" />
      </div>
      <p className="source">Цветом — сценарий только на [USR].</p>
    </div>
  )
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
        // PR-AUC случайного прогноза — доля ушедших.
        const chance = block.positives / block.rows
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
            <ul className="metrics">
              <li>
                <b>PR-AUC</b> — насколько хорошо модель находит ушедших. 1 — идеально, случайный прогноз — {chance.toFixed(3)} (доля ушедших).
              </li>
              <li>
                <b>ROC-AUC</b> — как часто ушедший получает балл выше оставшегося. 1 — всегда, 0.5 — наугад.
              </li>
              <li>
                <b>log-loss</b> — насколько верны сами вероятности. Меньше — лучше.
              </li>
            </ul>
            <p className="source">{`${block.rows} клиентов, ${block.positives} ушедших. Источник: ${scenarios.source}.`}</p>
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
    <div className="screen">
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
        </div>
      </div>

      {Object.entries(demo.downstream.scenarios?.tasks ?? {}).map(([task, block]) => (
        <ScenarioChart key={task} task={task} block={block} />
      ))}
    </div>
  )
}
