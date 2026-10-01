import { useMemo } from 'react'
import { attention } from '../../data/checkpoint'
import { demo, hero, num } from '../../data/demo'
import type { Vec3 } from '../../content/layout'
import { color as C } from '../../theme'
import { Appear, Arc, Block, Glyph, Label, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 10. HISTORY ENCODER: [USR] СОБИРАЕТ ИСТОРИЮ
// ============================================================
//
// Показаны 24 события из 617 (какие и с какими весами — в
// data/checkpoint.ts): с чекпойнтом толщина линии — настоящий вес
// внимания [USR] одной головы, без него — схема.
// ============================================================

const USR_X = -13.2
const FIRST_X = -10.4
const LAST_X = 12.6
const Y = -0.4

function Content() {
  const { beat } = useRegionView()
  const view = useMemo(attention, [])
  const events = view.events.map((event, i) => ({
    ...event,
    x: FIRST_X + (i / (view.events.length - 1)) * (LAST_X - FIRST_X),
  }))
  const config = demo.architecture.encoders.history.config

  // Типы самых весомых событий головы: на что она смотрит.
  const kinds = new Map<string, number>()
  for (const index of view.heaviest) {
    const type = demo.client.timeline[index].type
    kinds.set(type, (kinds.get(type) ?? 0) + 1)
  }
  const heaviest = [...kinds].sort((a, b) => b[1] - a[1]).map(([type, count]) => `${type} ×${count}`)

  const usr: Vec3 = [USR_X, Y + 1.1, 0.1]

  return (
    <group>
      <Label size={0.42} color={C.text} position={[-14.6, 6.0, 0]}>
        {`${demo.architecture.encoders.history.blocks} блока · ${config.heads} головы × ${demo.architecture.dim / Number(config.heads)} · FFN ${config.feedforward} · внимание в обе стороны`}
      </Label>

      <Block size={[28.4, 4.6, 1.2]} position={[-0.4, Y + 1.1, -0.4]} accent={C.cyan} faceOpacity={0.025} edgeOpacity={0.3} />
      <Block size={[28.8, 5.0, 1.6]} position={[-0.4, Y + 1.1, -0.9]} accent={C.cyan} faceOpacity={0.015} edgeOpacity={0.18} />

      {/* Слот [USR] */}
      <Glyph seed={4} accent={C.amber} position={[USR_X, Y + 1.1, 0]} height={2.6} width={0.42} caption={null} label="[USR]" />

      {/* События */}
      {events.map((event) => (
        <group key={event.index} position={[event.x, Y + 1.1, 0]}>
          <Glyph
            seed={event.index + 101}
            height={event.index === hero.index ? 2.6 : 1.9}
            width={event.index === hero.index ? 0.36 : 0.24}
            cells={event.index === hero.index ? 16 : 10}
            caption={null}
          />
        </group>
      ))}

      <Label mono size={0.3} color={C.text2} position={[FIRST_X - 0.2, Y - 1.1, 0]}>
        {view.real
          ? `${events.length} из ${demo.client.n_events} событий: самые весомые для [USR] и выборка · от старых к новым`
          : `${events.length} из ${demo.client.n_events} событий · от старых к новым`}
      </Label>

      {/* Внимание [USR] к событиям */}
      <Appear show={beat >= 1} speed={4}>
        {events.map((event) => (
          <Arc
            key={event.index}
            from={[usr[0], usr[1] + 1.4, usr[2]]}
            to={[event.x, Y + 2.5, 0]}
            lift={1.2 + Math.abs(event.x - usr[0]) * 0.14}
            accent={C.amber}
            width={0.6 + 2.8 * event.weight}
            opacity={0.12 + 0.78 * event.weight}
          />
        ))}
        <Label size={0.3} color={C.text2} position={[1.5, 5.75, 0]}>
          {view.real
            ? `толщина — вес внимания [USR]: блок ${view.block + 1}, голова ${view.head + 1} из ${config.heads}, самый большой ${num(view.max, 3)}`
            : 'толщина линии — вес внимания [USR] (голова 1 из 4, схема)'}
        </Label>
        {view.real ? (
          <Label size={0.3} color={C.text2} position={[1.5, 5.3, 0]}>
            {`${view.heaviest.length} самых весомых: ${heaviest.join(', ')}`}
          </Label>
        ) : null}
      </Appear>

      <Appear show={beat >= 2} delay={0.2} position={[-14.6, -3.4, 0]}>
        <Label size={0.38} color={C.amber}>
          {'[USR] после History Encoder → Client Embedding [128]'}
        </Label>
        <Label size={0.34} color={C.text2} position={[0, -0.6, 0]}>
          {'слоты событий → векторы событий с учётом всей истории'}
        </Label>
      </Appear>
    </group>
  )
}

export function HistoryEncoderRegion() {
  return (
    <Region id="historyEncoder" size={[32, 13]}>
      <Content />
    </Region>
  )
}
