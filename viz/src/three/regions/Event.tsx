import { hero } from '../../data/demo'
import { color as C } from '../../theme'
import { Appear, Card, Chip, Label, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 2. ОДНО СОБЫТИЕ: СМЫСЛОВЫЕ ПАРЫ КЛЮЧ = ЗНАЧЕНИЕ
// ============================================================

// Первые четыре — главные поля покупки, остальные — приглушённо.
export const MAIN_KEYS = ['event_type', 'transaction_amount', 'currency', 'merchant_name']

function rows(): { key: string; value: string; main: boolean }[] {
  const raw = hero.raw
  const main = MAIN_KEYS.filter((key) => key in raw).map((key) => ({ key, value: String(raw[key]), main: true }))
  const rest = Object.keys(raw)
    .filter((key) => !MAIN_KEYS.includes(key))
    .map((key) => ({ key, value: String(raw[key]), main: false }))
  return [...main, ...rest]
}

function Content() {
  const { beat } = useRegionView()
  const list = rows()
  const top = 4.6
  const step = 0.64

  return (
    <group>
      <Card width={17} height={list.length * step + 4.0} position={[-1.5, top - (list.length * step) / 2 + 0.1, -0.05]} />

      <Label size={0.78} color={C.cyan} position={[-9.2, top + 0.9, 0]}>
        {hero.type}
      </Label>
      <Label mono size={0.34} color={C.text2} position={[-9.2, top + 0.2, 0]}>
        {`источник ${hero.source}`}
      </Label>

      {list.map((row, index) => (
        <Appear key={row.key} show={beat >= 1} delay={0.06 * index} position={[0, top - 0.6 - index * step, 0]}>
          <Label mono size={0.4} anchorX="right" color={row.main ? C.text2 : C.muted} position={[-2.2, 0, 0]}>
            {row.key}
          </Label>
          <Label mono size={0.4} anchorX="center" color={C.muted} position={[-1.6, 0, 0]}>
            {'='}
          </Label>
          <Label mono size={0.4} color={row.main ? C.text : C.text2} position={[-1.0, 0, 0]}>
            {row.value}
          </Label>
        </Appear>
      ))}

      {/* Время — отдельный канал, не поле события. */}
      <Appear show={beat >= 1} delay={0.9} position={[11.2, 2.2, 0]}>
        <Chip text={hero.local_time.slice(0, 16)} accent={C.violet} strong size={0.4} />
        <Label size={0.36} color={C.violet} anchorX="center" position={[0, -1.05, 0]}>
          {'время — отдельный канал'}
        </Label>
        <Label size={0.32} color={C.text2} anchorX="center" position={[0, -1.6, 0]}>
          {'календарь и давность до T'}
        </Label>
      </Appear>
    </group>
  )
}

export function EventRegion() {
  return (
    <Region id="event" size={[30, 14]}>
      <Content />
    </Region>
  )
}
