import { demo, hero, num } from '../../data/demo'
import { color as C } from '../../theme'
import { Appear, Arrow, Block, Dial, Glyph, Label, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 6. КАЛЕНДАРЬ: 6 ЧИСЕЛ → MLP → [128], ПОСЛЕ ENCODER
// ============================================================

const WEEKDAYS = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']

// Угол точки на окружности по паре (sin, cos) из данных.
function angle(sin: number, cos: number): number {
  return Math.atan2(sin, cos)
}

function Content() {
  const { beat } = useRegionView()
  const [hs, hc, ds, dc, ms, mc] = hero.calendar
  const cycles = demo.dataset.calendar.cycles

  // Значения обратно из угла: номер дня недели от понедельника.
  const turn = (sin: number, cos: number, cycle: number) => {
    const a = angle(sin, cos)
    return (((a < 0 ? a + 2 * Math.PI : a) / (2 * Math.PI)) * cycle) % cycle
  }

  const hour = turn(hs, hc, cycles.hour_of_day)
  const weekday = Math.round(turn(ds, dc, cycles.day_of_week)) % 7
  const monthDay = Math.round(turn(ms, mc, cycles.day_of_month)) + 1

  const dials = [
    { a: angle(hs, hc), text: `час ${num(hour, 1)} / 24` },
    { a: angle(ds, dc), text: `${WEEKDAYS[weekday]} / 7` },
    { a: angle(ms, mc), text: `${monthDay}-е число / 31` },
  ]

  const names = demo.dataset.calendar.features

  return (
    <group>
      <Label size={0.46} color={C.violet} position={[-14.5, 5.8, 0]}>
        {`${hero.local_time.slice(0, 16)} · время банка`}
      </Label>

      {dials.map((dial, index) => (
        <Dial key={index} radius={1.35} angle={dial.a} position={[-12.8 + index * 3.6, 2.4, 0]} label={dial.text} />
      ))}

      <group position={[-14.5, -1.6, 0]}>
        {names.map((name, index) => (
          <Label key={name} mono size={0.32} color={C.text2} position={[(index % 3) * 3.6, -Math.floor(index / 3) * 0.6, 0]}>
            {`${name.replace('day_of_week', 'dow').replace('day_of_month', 'dom')} ${num(hero.calendar[index], 3)}`}
          </Label>
        ))}
      </group>

      <Appear show={beat >= 1}>
        <Arrow from={[-3.2, 1.2, 0]} to={[-1.4, 1.2, 0]} accent={C.violet} />
        <Block size={[3.4, 3.0, 1.2]} position={[0.6, 1.2, 0]} accent={C.violet} />
        <Label size={0.34} anchorX="center" position={[0.6, 1.8, 0.7]}>
          {'Linear 6→128'}
        </Label>
        <Label size={0.34} anchorX="center" position={[0.6, 1.2, 0.7]}>
          {'GELU'}
        </Label>
        <Label size={0.34} anchorX="center" position={[0.6, 0.6, 0.7]}>
          {'Linear 128→128'}
        </Label>
        <Arrow from={[2.5, 1.2, 0]} to={[3.7, 1.2, 0]} accent={C.violet} />
        <Glyph seed={311} accent={C.violet} position={[4.3, 1.2, 0]} height={3.2} width={0.36} label="календарь" caption="[128]" />
        <Label size={0.8} anchorX="center" color={C.text2} position={[5.7, 1.2, 0]}>
          {'+'}
        </Label>
        <Glyph seed={hero.index + 101} position={[7.1, 1.2, 0]} height={3.2} width={0.36} label="событие" caption="[128]" />
        <Label size={0.8} anchorX="center" color={C.text2} position={[8.5, 1.2, 0]}>
          {'='}
        </Label>
        <Glyph seed={hero.index + 700} position={[9.9, 1.2, 0]} height={3.2} width={0.36} label="датированное" caption="[128]" />
        <Label size={0.34} color={C.text2} position={[-3.2, -3.2, 0]}>
          {'прибавляется после Event Encoder к вектору [EVT] — не к токенам'}
        </Label>
      </Appear>
    </group>
  )
}

export function CalendarRegion() {
  return (
    <Region id="calendar" size={[32, 13]}>
      <Content />
    </Region>
  )
}
