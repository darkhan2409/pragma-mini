import { useFrame } from '@react-three/fiber'
import { useRef } from 'react'
import * as THREE from 'three'
import { demo, hero, num } from '../../data/demo'
import type { Vec3 } from '../../content/layout'
import { color as C } from '../../theme'
import { Appear, Dial, Label, Wire, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 9. TIMEROPE: ПОВОРОТ Q И K НА ДАВНОСТЬ ДО T
// ============================================================
//
// Углы настоящие: θ = позиция · 10000^(−2i/32) для плоскости i,
// позиции — из данных. Направления Q и K до поворота — схема.
// ============================================================

const BASE = 10000
const HEAD_DIM = demo.architecture.dim / Number(demo.architecture.encoders.history.config.heads)
const PLANE = 8

export const FREQUENCY = Math.pow(BASE, (-2 * PLANE) / HEAD_DIM)

const last = demo.client.timeline[demo.client.timeline.length - 1]

const rows = [
  { name: '[USR]', position: 0, accent: C.amber, q: 0.9, k: 2.2 },
  { name: `${hero.type} · ${String(hero.raw.merchant_name)}`, position: hero.time_log, accent: C.cyan, q: 0.5, k: 1.7 },
  { name: `${last.type} · последнее`, position: last.time_log, accent: C.cyan, q: 1.2, k: 2.6 },
]

function Rotating({ base, theta, rotate, accent, position, label }: { base: number; theta: number; rotate: boolean; accent: string; position: Vec3; label: string }) {
  const ref = useRef<THREE.Group>(null)
  const angle = useRef(base)

  useFrame((_, delta) => {
    angle.current = THREE.MathUtils.damp(angle.current, rotate ? base + theta : base, 2.2, Math.min(delta, 0.1))
    if (ref.current) ref.current.rotation.z = angle.current - base
  })

  return (
    <group position={position}>
      <Dial radius={1.15} angle={base} ghost={base} accent={C.muted} label={label} />
      <group ref={ref}>
        <Wire points={[[0, 0, 0.02], [1.15 * Math.cos(base), 1.15 * Math.sin(base), 0.02]]} accent={accent} width={2.4} opacity={1} />
      </group>
    </group>
  )
}

function Content() {
  const { beat } = useRegionView()
  const rotate = beat >= 1

  return (
    <group>
      <Label size={0.4} color={C.text} position={[-14.6, 6.0, 0]}>
        {'qkv: Linear 128 → 384 без bias · 4 головы по 32'}
      </Label>
      <Label size={0.32} color={C.text2} position={[-14.6, 5.4, 0]}>
        {`показана плоскость частоты ${PLANE} из 16: 10000^(−${2 * PLANE}/${HEAD_DIM}) = ${num(FREQUENCY, 3)}`}
      </Label>

      {['Q', 'K', 'V'].map((name, index) => (
        <Label key={name} size={0.5} anchorX="center" color={name === 'V' ? C.text2 : C.violet} position={[-3.6 + index * 3.8, 4.4, 0]}>
          {name}
        </Label>
      ))}

      {rows.map((row, index) => {
        const y = 2.5 - index * 3.1
        const theta = row.position * FREQUENCY
        return (
          <group key={row.name} position={[0, y, 0]}>
            <Label mono size={0.32} color={row.accent} position={[-14.6, 0.25, 0]}>
              {row.name}
            </Label>
            <Label mono size={0.3} color={C.text2} position={[-14.6, -0.35, 0]}>
              {`позиция ${num(row.position, 1)} · θ = ${num(theta, 3)} рад`}
            </Label>
            <Rotating base={row.q} theta={theta} rotate={rotate} accent={C.violet} position={[-3.6, 0, 0]} label="" />
            <Rotating base={row.k} theta={theta} rotate={rotate} accent={C.violet} position={[0.2, 0, 0]} label="" />
            <Dial radius={1.15} angle={row.k + 1.1} accent={C.text2} position={[4.0, 0, 0]} />
          </group>
        )
      })}

      <Appear show={rotate} position={[6.4, 1.2, 0]}>
        <Label size={0.36} color={C.text}>
          {"Q' = R(θ)·Q,  K' = R(θ)·K"}
        </Label>
        <Label size={0.34} color={C.text2} position={[0, -0.6, 0]}>
          {'V не поворачивается'}
        </Label>
        <Label size={0.34} color={C.text2} position={[0, -1.4, 0]}>
          {"q'·k' зависит от разности позиций:"}
        </Label>
        <Label size={0.34} color={C.text2} position={[0, -1.95, 0]}>
          {'насколько события далеки во времени'}
        </Label>
        <Label size={0.34} color={C.violet} position={[0, -2.9, 0]}>
          {'время — не слагаемое вектора, а поворот'}
        </Label>
      </Appear>
    </group>
  )
}

export function RopeRegion() {
  return (
    <Region id="rope" size={[32, 13]}>
      <Content />
    </Region>
  )
}
