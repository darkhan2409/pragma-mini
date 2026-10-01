import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from 'recharts'
import { kpis, progress, thin, type Telemetry } from '../../telemetry/model'
import { FileReader_, LiveReader, MockReader, listRuns, type RunEntry } from '../../telemetry/sources'
import { useTelemetry, type Reader } from '../../telemetry/useTelemetry'

// ============================================================
// ШАГ 14. ЖИВОЙ ДАШБОРД ОБУЧЕНИЯ
// ============================================================
//
// Те же показатели и та же семантика, что у Streamlit-страницы
// (src/dashboard/app.py): val — только настоящие точки после
// эпох, без значений между ними; одна ось на график; легенда —
// только где серий две. Цвета — ступени акцентов, проверенные
// валидатором dataviz для тёмной подложки.
// ============================================================

const SERIES = {
  train: '#0DA4BB',
  val: '#BF8601',
  clip: '#F24F6F',
  muted: '#5B6672',
}

const GRID = '#1E2733'

// Круглые деления оси шагов: 0, 5 000, 10 000…
function stepTicks(last: number): number[] {
  if (last <= 0) return [0]
  const raw = last / 4
  const power = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((m) => m >= raw) ?? raw
  const out: number[] = []
  for (let value = 0; value <= last; value += step) out.push(value)
  return out
}
const TICK = { fill: '#9AA5B1', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }

declare global {
  interface Window {
    showOpenFilePicker?: (options?: unknown) => Promise<FileSystemFileHandle[]>
  }
}

type Source = { kind: 'live'; run: string } | { kind: 'file'; name: string } | { kind: 'mock' }

function StepTooltip({ active, payload, label }: TooltipContentProps<number, string>) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: '#0D1117', border: '1px solid #1E2733', borderRadius: 8, padding: '8px 10px', fontSize: 12 }}>
      <div style={{ color: '#9AA5B1', marginBottom: 4 }}>шаг {Math.round(Number(label))}</div>
      {payload.map((item) => (
        <div key={String(item.name)} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 12, height: 2, background: String(item.color ?? item.stroke ?? '#9AA5B1'), display: 'inline-block' }} />
          <b style={{ color: '#E6EBF1', fontFamily: 'JetBrains Mono, monospace', fontWeight: 500 }}>
            {typeof item.value === 'number' ? (Math.abs(item.value) < 0.01 ? item.value.toExponential(2) : item.value.toFixed(4)) : String(item.value)}
          </b>
          <span style={{ color: '#9AA5B1' }}>{String(item.name)}</span>
        </div>
      ))}
    </div>
  )
}

function Tile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="tile">
      <div className="k">{label}</div>
      <div className="v">{value}</div>
      {hint ? <div className="h">{hint}</div> : null}
    </div>
  )
}

function status(telemetry: Telemetry | null): string {
  if (!telemetry || telemetry.modified === null) return 'файла телеметрии нет: в этом каталоге обучение ещё не начиналось'
  const where = progress(telemetry)
  const age = Date.now() / 1000 - telemetry.modified
  const parts = [`последняя запись ${Math.round(age)} с назад`]
  if (where.epochs !== null && where.completedEpochs >= where.epochs) parts.push('все эпохи пройдены')
  else if (age > 120 && telemetry.steps.size) parts.push('шагов давно нет: идёт val, обучение остановлено или упало')
  if (where.resumes) parts.push(`продолжений ${where.resumes}`)
  if (telemetry.skipped) parts.push(`битых строк пропущено ${telemetry.skipped}`)
  return parts.join(' · ')
}

export default function TrainingDashboard() {
  const [runs, setRuns] = useState<RunEntry[]>([])
  const [source, setSource] = useState<Source | null>(null)
  const [reader, setReader] = useState<Reader | null>(null)
  const [refresh, setRefresh] = useState(3)
  const [smoothing, setSmoothing] = useState(50)
  const [apiError, setApiError] = useState<string | null>(null)

  useEffect(() => {
    listRuns()
      .then((list) => {
        setRuns(list)
        const preferred = list.find((item) => item.id === 'runs/w4-b0' && item.telemetry) ?? list.find((item) => item.telemetry)
        if (preferred) {
          setSource({ kind: 'live', run: preferred.id })
          setReader(new LiveReader(preferred.id))
        }
      })
      .catch((error) => setApiError(`API недоступно (${String(error)}): откройте файл telemetry.jsonl`))
  }, [])

  const snapshot = useTelemetry(reader, refresh)
  const telemetry = snapshot.telemetry

  const view = useMemo(() => {
    if (!telemetry) return null
    const steps = [...telemetry.steps.values()].sort((a, b) => a.step - b.step)
    const window: number[] = []
    let sum = 0
    const rows = steps.map((item) => {
      window.push(item.loss)
      sum += item.loss
      if (window.length > smoothing) sum -= window.shift()!
      return {
        step: item.step,
        loss: smoothing > 1 ? sum / window.length : item.loss,
        lr: item.lr,
        grad_norm: item.grad_norm,
        tokens_per_second: item.seconds > 0 ? item.tokens / item.seconds : NaN,
      }
    })
    const thinned = thin(rows, 1500, ['loss', 'lr', 'grad_norm', 'tokens_per_second'])
    const epochs = [...telemetry.epochs.entries()].sort((a, b) => a[0] - b[0])
    const val = epochs.filter(([, record]) => record.val_loss !== null && record.val_loss !== undefined).map(([epoch, record]) => ({ step: record.step, val: record.val_loss as number, epoch }))
    const memory = epochs
      .filter(([, record]) => record.cuda_peak_allocated_gib !== null && record.cuda_peak_allocated_gib !== undefined)
      .map(([epoch, record]) => ({ epoch: `эп. ${epoch}`, allocated: record.cuda_peak_allocated_gib, reserved: record.cuda_peak_reserved_gib }))
    const run = telemetry.runs.length ? telemetry.runs[telemetry.runs.length - 1] : null
    const lastStep = steps.length ? steps[steps.length - 1].step : 0
    return { rows: thinned, val, memory, epochs, clip: run?.max_grad_norm ?? null, where: progress(telemetry), k: kpis(telemetry), ticks: stepTicks(lastStep), lastStep }
  }, [telemetry, snapshot.version, smoothing])

  const openFile = async () => {
    if (window.showOpenFilePicker) {
      const [handle] = await window.showOpenFilePicker({ types: [{ description: 'telemetry', accept: { 'application/json': ['.jsonl'] } }] })
      const next = new FileReader_(handle)
      setSource({ kind: 'file', name: next.name })
      setReader(next)
      return
    }
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.jsonl'
    input.onchange = () => {
      const file = input.files?.[0]
      if (!file) return
      const next = new FileReader_(file)
      setSource({ kind: 'file', name: next.name })
      setReader(next)
    }
    input.click()
  }

  const progressShare = view && view.where.totalSteps ? Math.min(1, view.where.step / view.where.totalSteps) : 0

  return (
    <div className="screen">
      <h1>Обучение</h1>
      <p className="sub">Живая телеметрия прогона — telemetry.jsonl каталога, дочитывается раз в {refresh} с.</p>

      <div className="controls">
        <span>прогон</span>
        <select
          value={source?.kind === 'live' ? source.run : ''}
          onChange={(event) => {
            const id = event.target.value
            setSource({ kind: 'live', run: id })
            setReader(new LiveReader(id))
          }}
        >
          {source?.kind !== 'live' ? <option value="">—</option> : null}
          {runs.map((run) => (
            <option key={run.id} value={run.id}>
              {run.id}
              {run.telemetry ? '' : ' (нет телеметрии)'}
            </option>
          ))}
        </select>
        <button className="linkbtn" style={{ marginTop: 0 }} onClick={openFile}>
          файл…
        </button>
        {import.meta.env.DEV ? (
          <button
            className="linkbtn"
            style={{ marginTop: 0 }}
            onClick={() => {
              setSource({ kind: 'mock' })
              setReader(new MockReader())
            }}
          >
            синтетика
          </button>
        ) : null}
        <span style={{ marginLeft: 12 }}>обновление</span>
        <select value={refresh} onChange={(event) => setRefresh(Number(event.target.value))}>
          {[2, 3, 5].map((value) => (
            <option key={value} value={value}>
              {value} с
            </option>
          ))}
        </select>
        <span style={{ marginLeft: 12 }}>сглаживание train loss</span>
        <select value={smoothing} onChange={(event) => setSmoothing(Number(event.target.value))}>
          {[1, 20, 50, 200].map((value) => (
            <option key={value} value={value}>
              {value === 1 ? 'нет' : `${value} шагов`}
            </option>
          ))}
        </select>
      </div>

      {source?.kind === 'mock' ? <div className="banner">СИНТЕТИКА — сгенерированные числа для проверки экрана, не результаты обучения.</div> : null}
      {apiError ? <div className="banner">{apiError}</div> : null}
      {snapshot.error ? <div className="banner">{snapshot.error}</div> : null}

      {view ? (
        <>
          <div className="tiles">
            <Tile label="Эпоха" value={view.k.epoch} />
            <Tile label="Шаг" value={view.k.step} />
            <Tile label="Train loss" value={view.k.trainLoss} hint="последний шаг" />
            <Tile label="Val loss" value={view.k.valLoss} hint={`лучший ${view.k.bestValLoss}`} />
            <Tile label="LR" value={view.k.lr} />
            <Tile label="Норма градиента" value={view.k.gradNorm} hint="до клипа" />
            <Tile label="Токенов/с" value={view.k.tokensPerSecond} hint="медиана 20 шагов" />
            <Tile label="Пик VRAM" value={view.k.vram} hint={`reserved ${view.k.vramReserved}`} />
          </div>
          <div className="progress">
            <div style={{ width: `${progressShare * 100}%` }} />
          </div>
          <p className="status-line">
            {source?.kind === 'live' ? source.run : source?.kind === 'file' ? source.name : 'синтетика'} · {status(telemetry)}
          </p>

          <div className="charts">
            <div className="card wide">
              <h3>LOSS ПО ШАГАМ ОПТИМИЗАТОРА · val — только после эпох</h3>
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart margin={{ top: 8, right: 24, bottom: 4, left: 4 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="step" type="number" domain={[0, view.lastStep]} ticks={view.ticks} tick={TICK} stroke={GRID} allowDuplicatedCategory={false} />
                  <YAxis tick={TICK} stroke={GRID} domain={['auto', 'auto']} width={48} />
                  <Tooltip content={(props) => <StepTooltip {...(props as TooltipContentProps<number, string>)} />} />
                  <Legend wrapperStyle={{ fontSize: 12, color: '#9AA5B1' }} />
                  <Line data={view.rows} dataKey="loss" name={smoothing > 1 ? `train (среднее ${smoothing})` : 'train'} stroke={SERIES.train} strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Scatter data={view.val} dataKey="val" name="val (после эпохи)" fill={SERIES.val} stroke="#0D1117" strokeWidth={2} isAnimationActive={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <h3>LEARNING RATE</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={view.rows} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="step" type="number" domain={[0, view.lastStep]} ticks={view.ticks} tick={TICK} stroke={GRID} />
                  <YAxis tick={TICK} stroke={GRID} width={56} tickFormatter={(value: number) => value.toExponential(0)} />
                  <Tooltip content={(props) => <StepTooltip {...(props as TooltipContentProps<number, string>)} />} />
                  <Line dataKey="lr" name="lr" stroke={SERIES.train} strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <h3>НОРМА ГРАДИЕНТА ДО КЛИПА{view.clip !== null ? ` · порог ${view.clip}` : ''}</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={view.rows} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="step" type="number" domain={[0, view.lastStep]} ticks={view.ticks} tick={TICK} stroke={GRID} />
                  <YAxis tick={TICK} stroke={GRID} width={40} domain={[0, 'auto']} allowDecimals tickCount={5} />
                  <Tooltip content={(props) => <StepTooltip {...(props as TooltipContentProps<number, string>)} />} />
                  {view.clip !== null ? <ReferenceLine y={view.clip} stroke={SERIES.clip} strokeDasharray="4 4" /> : null}
                  <Line dataKey="grad_norm" name="норма" stroke={SERIES.train} strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <h3>ТОКЕНОВ В СЕКУНДУ</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={view.rows} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="step" type="number" domain={[0, view.lastStep]} ticks={view.ticks} tick={TICK} stroke={GRID} />
                  <YAxis tick={TICK} stroke={GRID} width={64} tickFormatter={(value: number) => `${Math.round(value / 1000)}k`} />
                  <Tooltip content={(props) => <StepTooltip {...(props as TooltipContentProps<number, string>)} />} />
                  <Line dataKey="tokens_per_second" name="токенов/с" stroke={SERIES.train} strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <h3>ПИК ПАМЯТИ CUDA ПО ЭПОХАМ, ГиБ</h3>
              {view.memory.length ? (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={view.memory} margin={{ top: 8, right: 16, bottom: 4, left: 4 }} barGap={2}>
                    <CartesianGrid stroke={GRID} vertical={false} />
                    <XAxis dataKey="epoch" tick={TICK} stroke={GRID} />
                    <YAxis tick={TICK} stroke={GRID} width={40} />
                    <Tooltip cursor={{ fill: 'rgba(255,255,255,0.03)' }} contentStyle={{ background: '#0D1117', border: '1px solid #1E2733', fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 12, color: '#9AA5B1' }} />
                    <Bar dataKey="allocated" name="allocated" fill={SERIES.train} maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                    <Bar dataKey="reserved" name="reserved" fill={SERIES.muted} maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="status-line">пики появятся после первой эпохи на CUDA</p>
              )}
            </div>

            <div className="card wide">
              <h3>ЭПОХИ</h3>
              <table className="grid">
                <thead>
                  <tr>
                    <th>эпоха</th>
                    <th>шаг</th>
                    <th>train loss</th>
                    <th>val loss</th>
                    <th>LR</th>
                    <th>train, мин</th>
                    <th>val, мин</th>
                    <th>норма ср.</th>
                    <th>клип</th>
                    <th>VRAM</th>
                  </tr>
                </thead>
                <tbody>
                  {view.epochs.map(([epoch, record]) => (
                    <tr key={epoch}>
                      <td className="num">{epoch}</td>
                      <td className="num">{record.step}</td>
                      <td className="num">{record.train_loss?.toFixed(4) ?? '—'}</td>
                      <td className="num">{record.val_loss?.toFixed(4) ?? '—'}</td>
                      <td className="num">{record.learning_rate?.toExponential(2) ?? '—'}</td>
                      <td className="num">{record.train_seconds !== undefined ? (record.train_seconds / 60).toFixed(1) : '—'}</td>
                      <td className="num">{record.val_seconds !== undefined ? (record.val_seconds / 60).toFixed(1) : '—'}</td>
                      <td className="num">{record.grad_norm_mean?.toFixed(3) ?? '—'}</td>
                      <td className="num">{record.clipped_share !== null && record.clipped_share !== undefined ? `${Math.round(record.clipped_share * 100)}%` : '—'}</td>
                      <td className="num">{record.cuda_peak_allocated_gib?.toFixed(2) ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : (
        <p className="status-line">{apiError ?? 'загрузка телеметрии…'}</p>
      )}
    </div>
  )
}
