import { useMemo } from 'react'
import { demo, int, type CatBoostTraining, type LrTraining } from '../../data/demo'
import type { Vec3 } from '../../content/layout'
import { color as C } from '../../theme'
import { Appear, Arrow, Block, Card, Chip, Glyph, Label, Wire, chipWidth, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГИ 17 И 18. КАК УЧАТСЯ ГОЛОВЫ
// ============================================================
//
// Регрессия над [USR]: перебор C по фолдам train (кривая log-loss),
// финальная модель на всём train, порог по прогнозам фолдов, val
// только оценивает. CatBoost на handcrafted-признаках и на них же с
// [USR]: ранняя остановка по PR-AUC на отложенной части train (две
// кривые), там же порог, затем весь train с тем же числом деревьев.
// Все числа — из экспорта (export_demo.py --run); без него регион
// говорит, чего не хватает.
// ============================================================

// Две модели CatBoost — две серии; палитра проверена на тёмном фоне.
const SERIES: Record<string, { color: string; name: string; short: string }> = {
  catboost: { color: '#0DA4BB', name: 'handcrafted', short: 'handcrafted' },
  catboost_plus_usr: { color: '#BF8601', name: 'handcrafted + [USR]', short: '+ [USR]' },
}

const training = Object.values(demo.downstream.training?.tasks ?? {})[0]

function plural(n: number, forms: [string, string, string]): string {
  const tail = n % 100
  if (tail >= 11 && tail <= 14) return forms[2]
  if (n % 10 === 1) return forms[0]
  if (n % 10 >= 2 && n % 10 <= 4) return forms[1]
  return forms[2]
}

const trees = (n: number) => `${int(n)} ${plural(n, ['дерево', 'дерева', 'деревьев'])}`

function scaleC(c: number): string {
  if (c >= 1) return String(c)
  return `1e${Math.round(Math.log10(c))}`
}

// log-loss прогноза одной долей ушедших — уровень «без модели».
function constantLogLoss(rate: number): number {
  return -(rate * Math.log(rate) + (1 - rate) * Math.log(1 - rate))
}

function Dot({ position, radius = 0.12, color }: { position: Vec3; radius?: number; color: string }) {
  return (
    <mesh position={position} renderOrder={6}>
      <circleGeometry args={[radius, 24]} />
      <meshBasicMaterial color={color} />
    </mesh>
  )
}

function Missing() {
  return (
    <Label size={0.36} color={C.text2} position={[-14.6, 0, 0]} maxWidth={28}>
      {'В экспорте нет обучения голов: python -m src.downstream.probe --tag <прогон>, churn_baseline, затем export_demo.py --run'}
    </Label>
  )
}

// ------------------------------------------------------------
// Регрессия над [USR]
// ------------------------------------------------------------

function LrChart({ lr }: { lr: LrTraining }) {
  const chart = useMemo(() => {
    const x0 = -9.0
    const x1 = 1.8
    const y0 = -3.6
    const y1 = 1.6
    const flat = constantLogLoss(lr.train.positives / lr.train.rows)
    const values = [...lr.grid.map((point) => point.log_loss), flat]
    const pad = (Math.max(...values) - Math.min(...values)) * 0.12
    const lo = Math.min(...values) - pad
    const hi = Math.max(...values) + pad
    const logs = lr.grid.map((point) => Math.log10(point.C))
    const xAt = (c: number) => x0 + ((Math.log10(c) - logs[0]) / (logs[logs.length - 1] - logs[0])) * (x1 - x0)
    const yAt = (value: number) => y0 + ((value - lo) / (hi - lo)) * (y1 - y0)
    const points = lr.grid.map((point) => [xAt(point.C), yAt(point.log_loss), 0] as Vec3)
    return { x0, x1, y0, y1, flat, points, xAt, yAt }
  }, [lr])

  const best = lr.grid.find((point) => point.C === lr.C)?.log_loss ?? 0
  const chosen: Vec3 = [chart.xAt(lr.C), chart.yAt(best), 0.02]

  return (
    <group>
      <Label size={0.3} color={C.text2} position={[chart.x0, chart.y1 + 0.9, 0]}>
        {'log-loss на проверочном фолде · меньше — лучше'}
      </Label>
      <Label size={0.28} color={C.text} position={[chart.x0, chart.y1 + 0.3, 0]}>
        {`выбран C = ${lr.C}: log-loss ${best.toFixed(4)}`}
      </Label>
      <Wire points={[[chart.x0, chart.y0, 0], [chart.x1, chart.y0, 0]]} accent={C.muted} width={1} opacity={0.9} />
      <Wire points={[[chart.x0, chart.yAt(chart.flat), 0], [chart.x1, chart.yAt(chart.flat), 0]]} accent={C.muted} width={1} opacity={0.8} dashed />
      <Label mono size={0.26} color={C.muted} anchorX="right" position={[chart.x1, chart.yAt(chart.flat) + 0.3, 0]}>
        {`без модели (доля ушедших) ${chart.flat.toFixed(3)}`}
      </Label>
      <Wire points={chart.points} accent={C.cyan} width={2} opacity={0.95} />
      {chart.points.map((point, index) => (
        <group key={lr.grid[index].C}>
          <Dot position={[point[0], point[1], 0.01]} color={C.cyan} />
          <Label mono size={0.26} anchorX="center" anchorY="top" color={C.text2} position={[point[0], chart.y0 - 0.3, 0]}>
            {scaleC(lr.grid[index].C)}
          </Label>
        </group>
      ))}
      <Dot position={chosen} radius={0.22} color={C.amber} />
      <Label size={0.26} color={C.muted} position={[chart.x0, chart.y0 - 1.0, 0]}>
        {'сила регуляризации C'}
      </Label>
    </group>
  )
}

function LrContent() {
  const { beat } = useRegionView()
  const lr = training?.lr

  if (!lr) return <Missing />

  return (
    <group>
      <Label size={0.38} color={C.text} position={[-14.6, 5.9, 0]}>
        {`train: ${int(lr.train.rows)} клиентов × [USR] ${demo.architecture.dim} · ушли ${int(lr.train.positives)}`}
      </Label>

      {[0, 1, 2].map((index) => (
        <Glyph key={index} seed={41 + index} accent={C.amber} position={[-13.6 + index * 0.6, 1.2 - index * 0.2, -index * 0.05]} height={4.0} width={0.42} caption={null} />
      ))}
      <Label size={0.28} anchorX="center" color={C.text2} position={[-13.0, -1.6, 0]}>
        {'[USR] клиентов'}
      </Label>
      <Label size={0.3} color={C.muted} position={[-14.6, -3.0, 0]}>
        {`val: ${int(lr.val.rows)} клиентов`}
      </Label>
      <Label size={0.3} color={C.muted} position={[-14.6, -3.5, 0]}>
        {'только прогноз'}
      </Label>

      <Appear show={beat >= 1}>
        <Arrow from={[-11.4, 1.0, 0]} to={[-9.8, 1.0, 0]} accent={C.amber} />
        {Array.from({ length: lr.folds }, (_, index) => (
          <Chip key={index} text={`фолд ${index + 1}`} size={0.32} accent={C.cyan} position={[-7.6 + index * 2.5, 4.6, 0]} />
        ))}
        <Label size={0.26} color={C.text2} position={[-9.0, 3.7, 0]}>
          {'учится на двух, проверяется на третьем · StandardScaler внутри фолда'}
        </Label>
        <LrChart lr={lr} />
      </Appear>

      <Appear show={beat >= 2}>
        <Arrow from={[2.4, 0.4, 0]} to={[3.8, 0.4, 0]} accent={C.amber} />
        <Block size={[5.6, 2.8, 1]} position={[6.8, 0.4, 0]} accent={C.amber} />
        <Label size={0.32} anchorX="center" position={[6.8, 1.15, 0.6]}>
          {'StandardScaler'}
        </Label>
        <Label size={0.32} anchorX="center" position={[6.8, 0.5, 0.6]}>
          {'LogisticRegression'}
        </Label>
        <Label mono size={0.26} anchorX="center" color={C.text2} position={[6.8, -0.2, 0.6]}>
          {`C = ${lr.C} · весь train`}
        </Label>
        <Label size={0.28} anchorX="center" color={C.text2} position={[6.8, 2.6, 0]}>
          {`порог max F1 по прогнозам фолдов: ${lr.threshold.toFixed(3)}`}
        </Label>
      </Appear>

      <Appear show={beat >= 3}>
        <Arrow from={[6.8, -1.2, 0]} to={[6.8, -2.6, 0]} accent={C.amber} />
        <Label size={0.34} anchorX="center" color={C.text} position={[6.8, -3.2, 0]}>
          {`val: PR-AUC ${lr.val.pr_auc.toFixed(3)} · ROC-AUC ${lr.val.roc_auc.toFixed(3)} · F1 ${lr.val.f1.toFixed(3)}`}
        </Label>
        <Label size={0.28} anchorX="center" color={C.muted} position={[6.8, -3.85, 0]}>
          {'только прогноз — val в обучение не попадает'}
        </Label>
      </Appear>
    </group>
  )
}

export function LrTrainingRegion() {
  return (
    <Region id="lrTraining" size={[30, 13]}>
      <LrContent />
    </Region>
  )
}

// ------------------------------------------------------------
// CatBoost
// ------------------------------------------------------------

function CatBoostChart({ models }: { models: [string, CatBoostTraining][] }) {
  const chart = useMemo(() => {
    const x0 = -6.6
    const x1 = 2.4
    const y0 = -3.6
    const y1 = 2.4
    const last = Math.max(...models.map(([, item]) => item.curve[item.curve.length - 1][0]))
    const values = models.flatMap(([, item]) => item.curve.map((point) => point[1]))
    const pad = (Math.max(...values) - Math.min(...values)) * 0.08
    const lo = Math.min(...values) - pad
    const hi = Math.max(...values) + pad
    const xAt = (iteration: number) => x0 + (iteration / last) * (x1 - x0)
    const yAt = (value: number) => y0 + ((value - lo) / (hi - lo)) * (y1 - y0)
    const step = Math.pow(10, Math.floor(Math.log10(last / 3)))
    const tick = [1, 2, 5].map((k) => k * step).find((size) => last / size <= 4) ?? step * 10
    const ticks = Array.from({ length: Math.floor(last / tick) + 1 }, (_, i) => i * tick)
    return { x0, x1, y0, y1, xAt, yAt, ticks }
  }, [models])

  const [, first] = models[0]
  const wait = first.curve[first.curve.length - 1][0] - first.best_iteration

  return (
    <group>
      <Label size={0.3} color={C.text2} position={[chart.x0, chart.y1 + 1.0, 0]}>
        {'PR-AUC на отложенной части train · по числу деревьев'}
      </Label>
      <Wire points={[[chart.x0, chart.y0, 0], [chart.x1, chart.y0, 0]]} accent={C.muted} width={1} opacity={0.9} />
      {chart.ticks.map((value) => (
        <Label key={value} mono size={0.26} anchorX="center" anchorY="top" color={C.text2} position={[chart.xAt(value), chart.y0 - 0.3, 0]}>
          {String(value)}
        </Label>
      ))}
      {models.map(([name, item], index) => {
        const series = SERIES[name] ?? SERIES.catboost
        const best = item.curve.find((point) => point[0] === item.best_iteration) ?? item.curve[0]
        const end = item.curve[item.curve.length - 1]
        return (
          <group key={name}>
            <Wire points={item.curve.map(([x, y]) => [chart.xAt(x), chart.yAt(y), 0] as Vec3)} accent={series.color} width={2} opacity={0.95} />
            <Wire points={[[chart.xAt(best[0]), chart.y0, 0], [chart.xAt(best[0]), chart.yAt(best[1]), 0]]} accent={series.color} width={1} opacity={0.8} dashed />
            <Dot position={[chart.xAt(best[0]), chart.yAt(best[1]), 0.01]} radius={0.16} color={series.color} />
            <Label mono size={0.26} color={C.text} position={[chart.xAt(best[0]) + 0.12, chart.y0 + 0.3 + index * 0.4, 0]}>
              {String(item.best_iteration + 1)}
            </Label>
            <Chip text={series.short} size={0.26} accent={series.color} textColor={C.text} position={[chart.x1 + 0.3 + chipWidth(series.short, 0.26) / 2, chart.yAt(end[1]), 0]} />
          </group>
        )
      })}
      <Label size={0.26} color={C.muted} position={[chart.x0, chart.y0 - 1.0, 0]}>
        {`деревьев; пунктир — лучшее число, после него ещё ${wait} без улучшения — остановка`}
      </Label>
    </group>
  )
}

function CatBoostContent() {
  const { beat } = useRegionView()
  const models = Object.entries(training?.catboost ?? {})
  const first = models[0]?.[1]

  if (!first) return <Missing />

  const share = first.holdout_share
  const total = first.inner_train.rows + first.inner_holdout.rows
  const bar = 6.4
  const kept = bar * (first.inner_train.rows / total)

  return (
    <group>
      <Label size={0.36} color={C.text} position={[-14.6, 5.9, 0]}>
        {`train: ${int(total)} клиентов · ${Math.round(share * 100)}% отложено (seed ${demo.downstream.training?.seed}) · val не участвует`}
      </Label>

      {models.map(([name, item], index) => {
        const series = SERIES[name] ?? SERIES.catboost
        return (
          <Card key={name} width={6.4} height={0.9} position={[-11.4, 4.2 - index * 1.2, 0]} stroke={series.color} strokeOpacity={0.9}>
            <Label size={0.28} anchorX="center" color={C.text} position={[0, 0, 0.02]}>
              {`${series.name} · ${item.features} ${plural(item.features, ['признак', 'признака', 'признаков'])}`}
            </Label>
          </Card>
        )
      })}

      <Card width={kept - 0.05} height={0.8} radius={0.12} position={[-14.6 + kept / 2, 0.6, 0]} stroke={C.text2} strokeOpacity={0.7} />
      <Card width={bar - kept - 0.05} height={0.8} radius={0.12} position={[-14.6 + kept + (bar - kept) / 2, 0.6, 0]} stroke={C.amber} strokeOpacity={0.9} />
      <Label size={0.26} color={C.text2} position={[-14.6, -0.3, 0]}>
        {`учит: ${int(first.inner_train.rows)}`}
      </Label>
      <Label size={0.26} anchorX="right" color={C.text2} position={[-14.6 + bar, -0.3, 0]}>
        {`проверяет: ${int(first.inner_holdout.rows)}`}
      </Label>
      <Label size={0.26} color={C.muted} position={[-14.6, -0.85, 0]}>
        {'разбиение стратифицировано по уходу'}
      </Label>

      <Appear show={beat >= 1}>
        <CatBoostChart models={models} />
      </Appear>

      <Appear show={beat >= 2}>
        <Label size={0.28} color={C.text2} position={[7.0, 5.0, 0]}>
          {'порог max F1 на отложенной части'}
        </Label>
        {models.map(([name, item], index) => (
          <Label key={name} mono size={0.28} color={C.text} position={[7.0, 4.3 - index * 0.55, 0]}>
            {`${(SERIES[name] ?? SERIES.catboost).short}: ${item.threshold.toFixed(3)}`}
          </Label>
        ))}
      </Appear>

      <Appear show={beat >= 3}>
        <Label size={0.28} color={C.text2} position={[7.0, 2.2, 0]}>
          {'заново на всём train, столько же деревьев'}
        </Label>
        {models.map(([name, item], index) => (
          <Label key={name} mono size={0.28} color={C.text} position={[7.0, 1.5 - index * 0.55, 0]}>
            {`${(SERIES[name] ?? SERIES.catboost).short}: ${trees(item.trees)}`}
          </Label>
        ))}
        <Label size={0.28} color={C.text2} position={[7.0, -0.5, 0]}>
          {'val — только прогноз'}
        </Label>
        {models.map(([name, item], index) => (
          <Label key={name} mono size={0.28} color={C.text} position={[7.0, -1.2 - index * 0.55, 0]}>
            {`${(SERIES[name] ?? SERIES.catboost).short}: PR-AUC ${item.val.pr_auc.toFixed(3)} · ROC-AUC ${item.val.roc_auc.toFixed(3)}`}
          </Label>
        ))}
      </Appear>
    </group>
  )
}

export function CatBoostTrainingRegion() {
  return (
    <Region id="catboostTraining" size={[30, 13]}>
      <CatBoostContent />
    </Region>
  )
}
