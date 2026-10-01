import { demo, int, localDay } from '../../data/demo'
import { color as C } from '../../theme'
import { Appear, Arrow, Block, Card, Glyph, Label, Wire, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 15. CLIENT EMBEDDING БЕЗ MLM-ГОЛОВЫ
// ШАГ 16. ЗАДАЧИ НА ВЕКТОРЕ КЛИЕНТА
// ============================================================

function Embedding() {
  const { beat } = useRegionView()

  return (
    <group>
      <Glyph seed={4} accent={C.amber} position={[0, 1.2, 0]} height={7.2} width={0.9} caption={`[${demo.architecture.dim}]`} />
      <Label size={1.5} anchorX="center" color={C.amber} position={[0, 6.8, 0]}>
        {'Client Embedding'}
      </Label>
      <Appear show={beat >= 1} position={[0, -4.6, 0]}>
        <Label size={0.9} anchorX="center" color={C.text2}>
          {'[USR] после History Encoder'}
        </Label>
        <Label size={0.8} anchorX="center" color={C.muted} position={[0, -1.3, 0]}>
          {`энкодер: ${int(demo.architecture.total_parameters - demo.architecture.head_parameters)} параметров без головы`}
        </Label>
      </Appear>
    </group>
  )
}

export function ClientEmbeddingRegion() {
  return (
    <Region id="clientEmbedding" size={[18, 16]}>
      <Embedding />
    </Region>
  )
}

const TASK_TEXT: Record<string, [string, string]> = {
  churn_active90: ['churn_active90', 'нет действий в (T, T+60 дней] у активных за 90 дней'],
}

function Downstream() {
  const { beat } = useRegionView()
  const d = demo.downstream
  const catboost = d.catboost_churn

  return (
    <group>
      <Glyph seed={4} accent={C.amber} position={[-13.4, 1.4, 0]} height={4.6} width={0.5} caption={null} label="[USR] на T" />
      <Label size={0.32} color={C.text2} position={[-14.6, -1.6, 0]}>
        {`train: T = ${localDay(d.cutoffs.train)} — конец истории обучения`}
      </Label>
      <Label mono size={0.28} color={C.muted} position={[-14.6, -2.15, 0]}>
        {`val: T = ${localDay(d.cutoffs.val)} — конец выгрузки − ${d.horizon_days} дней`}
      </Label>

      {/* Наборы векторов */}
      <group position={[-9.6, 3.6, 0]}>
        <Label size={0.34} color={C.text2}>
          {'readouts + recency'}
        </Label>
        {d.readouts.map((name, index) => (
          <group key={name} position={[0.4 + index * 1.7, -2.2, 0]}>
            <Glyph seed={index + 20} accent={index < 2 ? C.amber : C.cyan} height={2.4} width={0.3} cells={10} caption={null} />
            <Label mono size={0.22} anchorX="center" color={C.text2} position={[0, -1.6, 0]}>
              {name}
            </Label>
          </group>
        ))}
      </group>
      <Arrow from={[-11.8, 1.4, 0]} to={[-10.2, 1.4, 0]} accent={C.amber} />

      <Arrow from={[-3.0, 1.4, 0]} to={[-2.4, 1.4, 0]} accent={C.amber} />
      <Block size={[4.6, 2.6, 1]} position={[0, 1.4, 0]} accent={C.amber} />
      <Label size={0.32} anchorX="center" position={[0, 1.8, 0.6]}>
        {'StandardScaler'}
      </Label>
      <Label size={0.32} anchorX="center" position={[0, 1.1, 0.6]}>
        {'LogisticRegression'}
      </Label>

      <Appear show={beat >= 1}>
        {d.tasks.map((task, index) => {
          const y = 4.8 - index * 2.4
          const [title, text] = TASK_TEXT[task] ?? [task, '']
          return (
            <group key={task}>
              <Wire points={[[2.4, 1.4, 0], [4.2, 1.4, 0], [4.2, y, 0], [5.4, y, 0]]} accent={C.amber} width={1.4} opacity={0.8} />
              <Card width={9.6} height={2.1} position={[10.4, y, -0.05]} stroke={C.amber} strokeOpacity={0.6} />
              <Label size={0.5} color={C.amber} position={[5.9, y + 0.35, 0]}>
                {title}
              </Label>
              <Label size={0.3} color={C.text2} position={[5.9, y - 0.4, 0]}>
                {text}
              </Label>
            </group>
          )
        })}

        {catboost ? (
          <group position={[10.4, -4.85, 0]}>
            <Card width={9.6} height={2.2} position={[0, 0, -0.05]} stroke={C.muted} strokeOpacity={0.7} fillOpacity={0.6} />
            <Label size={0.32} color={C.text} position={[-4.5, 0.55, 0]}>
              {'CatBoost — опорный бейзлайн churn'}
            </Label>
            <Label size={0.26} color={C.text2} position={[-4.5, -0.02, 0]}>
              {`на агрегатах, не на векторе · PR-AUC на ${catboost.group}`}
            </Label>
            <Label mono size={0.26} color={C.text2} position={[-4.5, -0.55, 0]}>
              {Object.entries(catboost.tasks)
                .map(([task, result]) => `${task} ${result.pr_auc.toFixed(3)}`)
                .join(' · ')}
            </Label>
          </group>
        ) : null}
      </Appear>
    </group>
  )
}

export function DownstreamRegion() {
  return (
    <Region id="downstream" size={[32, 14]}>
      <Downstream />
    </Region>
  )
}
