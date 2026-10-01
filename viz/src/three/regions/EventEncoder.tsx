import { useMemo } from 'react'
import { demo, type EventView } from '../../data/demo'
import { color as C } from '../../theme'
import { Appear, Arc, Arrow, Block, Glyph, Label, Wire, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 5. EVENT ENCODER: КАЖДОЕ СОБЫТИЕ ОТДЕЛЬНО → [EVT]
// ============================================================

const lanes: { event: EventView; y: number }[] = [
  { event: demo.client.highlighted.hero, y: 4.0 },
  { event: demo.client.highlighted.salary_credit, y: 0.6 },
  { event: demo.client.highlighted.app_screen, y: -2.8 },
]

const TOKENS_X = -13.2
const CELL = 0.38
const GAP = 0.1
const STACK_X = 0.4
const OUT_X = 9.2

// Дуги внимания внутри события — схема, не веса модели.
function arcsFor(count: number, seed: number): [number, number, number][] {
  const out: [number, number, number][] = []
  let s = seed
  for (let i = 0; i < Math.min(6, count - 1); i++) {
    s = (s * 1103515245 + 12345) % 2147483648
    const a = s % count
    s = (s * 1103515245 + 12345) % 2147483648
    const b = s % count
    if (a !== b) out.push([Math.min(a, b), Math.max(a, b), 0.3 + (s % 100) / 180])
  }
  return out
}

function Lane({ event, y, beat }: { event: EventView; y: number; beat: number }) {
  const count = event.tokens.length
  const x = (i: number) => TOKENS_X + i * (CELL + GAP)
  const arcs = useMemo(() => arcsFor(count, event.index + 7), [count, event.index])
  const end = x(count - 1) + CELL

  return (
    <group position={[0, y, 0]}>
      <Label mono size={0.36} color={C.text} position={[TOKENS_X, 1.0, 0]}>
        {`${event.type} · ${count} токенов`}
      </Label>

      {event.tokens.map((token, index) => (
        <mesh key={index} position={[x(index) + CELL / 2, 0, 0]}>
          <planeGeometry args={[CELL, CELL]} />
          <meshBasicMaterial color={index === 0 ? C.cyan : token.position > 0 ? '#2F5E6B' : '#24424C'} />
        </mesh>
      ))}

      <Appear show={beat >= 1}>
        {arcs.map(([a, b, w], index) => (
          <Arc
            key={index}
            from={[x(a) + CELL / 2, CELL / 2 + 0.05, 0]}
            to={[x(b) + CELL / 2, CELL / 2 + 0.05, 0]}
            lift={0.25 + (b - a) * 0.09}
            opacity={w}
            width={1}
          />
        ))}
        <Arrow from={[end + 0.4, 0, 0]} to={[STACK_X - 0.7, 0, 0]} accent={C.muted} />
        {Array.from({ length: demo.architecture.encoders.event.blocks }, (_, i) => (
          <Block key={i} size={[0.42, 2.0, 1.2]} position={[STACK_X + i * 0.95, 0, 0]} edgeOpacity={0.55} />
        ))}
      </Appear>

      <Appear show={beat >= 2} delay={0.1}>
        <Arrow from={[STACK_X + 4 * 0.95 + 0.6, 0, 0]} to={[OUT_X - 0.8, 0, 0]} accent={C.cyan} />
        <Glyph seed={event.index + 101} position={[OUT_X, 0, 0]} height={2.6} width={0.36} caption={null} />
        <Label size={0.34} color={C.cyan} position={[OUT_X + 0.7, 0.3, 0]}>
          {'вектор события'}
        </Label>
        <Label mono size={0.3} color={C.text2} position={[OUT_X + 0.7, -0.3, 0]}>
          {'[EVT] · [128]'}
        </Label>
      </Appear>
    </group>
  )
}

function Content() {
  const { beat } = useRegionView()
  const config = demo.architecture.encoders.event.config

  return (
    <group>
      {lanes.map((lane) => (
        <Lane key={lane.event.index} {...lane} beat={beat} />
      ))}

      {/* Границы событий: внимание их не пересекает. */}
      {[2.3, -1.1].map((y) => (
        <Wire key={y} points={[[TOKENS_X, y, 0], [OUT_X + 4.6, y, 0]]} accent={C.line} width={1} opacity={1} dashed />
      ))}

      <Appear show={beat >= 1} delay={0.3} position={[STACK_X - 0.3, 6.3, 0]}>
        <Label size={0.4} color={C.text}>
          {`${demo.architecture.encoders.event.blocks} блоков · ${config.heads} головы · FFN ${config.feedforward}`}
        </Label>
        <Label size={0.32} color={C.text2} position={[0, -0.55, 0]}>
          {'LN → внимание → + → LN → FFN → +'}
        </Label>
      </Appear>
      <Appear show={beat >= 1} delay={0.5} position={[TOKENS_X, -5.0, 0]}>
        <Label size={0.34} color={C.text2}>
          {'внимание не выходит за границы события · позиций и времени здесь нет'}
        </Label>
      </Appear>
    </group>
  )
}

export function EventEncoderRegion() {
  return (
    <Region id="eventEncoder" size={[34, 15]}>
      <Content />
    </Region>
  )
}
