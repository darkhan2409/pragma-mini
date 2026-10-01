import { demo, num } from '../../data/demo'
import { ANCHOR, type RegionId, type Vec3 } from '../../content/layout'
import { color as C } from '../../theme'
import { Appear, Arc, Label, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 12. LOSS И BACKWARD НА ОБЩЕМ ПЛАНЕ
// ============================================================
//
// Градиенты идут назад по тем же связям, что и данные: голова →
// история → анкета и события (с календарём) → таблица. Таблица
// получает градиент с двух сторон: как вход и как связанный
// выход головы. Толщина линий — схема, не величины градиентов.
// ============================================================

const HERE = ANCHOR.backprop

function rel(region: RegionId, dy = 0): Vec3 {
  const [x, y, z] = ANCHOR[region]
  return [x - HERE[0], y - HERE[1] + dy, z - HERE[2]]
}

const FLOWS: [RegionId, RegionId][] = [
  ['mlm', 'historyEncoder'],
  ['historyEncoder', 'history'],
  ['history', 'profile'],
  ['history', 'eventEncoder'],
  ['eventEncoder', 'calendar'],
  ['eventEncoder', 'embedding'],
  ['profile', 'embedding'],
]

function Content() {
  const { beat } = useRegionView()
  const run = demo.run
  const clipped = run?.epochs[0]?.clipped_share
  const t = demo.training

  return (
    <group>
      <group position={rel('mlm', 4.6)}>
        <Label size={2.2} anchorX="center" color={C.rose}>
          {'loss'}
        </Label>
        <Label size={1.3} anchorX="center" color={C.text2} position={[0, -2.2, 0]}>
          {`cross-entropy · сглаживание ${num(Number(t.label_smoothing), 1)} · среднее по целям`}
        </Label>
      </group>

      <Appear show={beat >= 1} speed={3}>
        {FLOWS.map(([from, to]) => (
          <Arc key={`${from}-${to}`} from={rel(from)} to={rel(to)} lift={4} accent={C.rose} width={2.2} opacity={0.9} flow={2.2} />
        ))}
        <Arc from={rel('mlm', 2)} to={rel('embedding', 2)} lift={22} accent={C.rose} width={1.6} opacity={0.7} dashed flow={1.6} />
        <Label size={1.3} anchorX="center" color={C.rose} position={[rel('embedding')[0] + 44, 24, 0]}>
          {'связанный выход головы: та же таблица E'}
        </Label>
        <group position={rel('embedding', -11)}>
          <Label size={1.5} anchorX="center" color={C.text}>
            {`AdamW · lr ${String(t.learning_rate)} · weight decay ${String(t.weight_decay)}`}
          </Label>
          <Label size={1.2} anchorX="center" color={C.text2} position={[0, -2.1, 0]}>
            {`clip ‖g‖ ≤ ${num(Number(t.max_grad_norm), 1)}${clipped !== null && clipped !== undefined ? ` · в эпохе 1 обрезано ${Math.round(clipped * 100)}% шагов (B0)` : ''}`}
          </Label>
        </group>
      </Appear>
    </group>
  )
}

export function BackpropRegion() {
  return (
    <Region id="backprop" size={[120, 60]}>
      <Content />
    </Region>
  )
}
