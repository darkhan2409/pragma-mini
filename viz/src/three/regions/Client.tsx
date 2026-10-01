import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { demo, hero, int, localDay, localMidnight } from '../../data/demo'
import type { Vec3 } from '../../content/layout'
import { color as C, sourceColor } from '../../theme'
import { Appear, Label, Wire, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 1. КЛИЕНТ И ЕГО СОБЫТИЯ НА ОСИ ВРЕМЕНИ
// ============================================================

const LEFT = -12.5
const RIGHT = 12.5
const AXIS_Y = -4.2
const LANE_Y0 = -3.6
const LANE_STEP = 0.66

const timeline = demo.client.timeline
const start = Date.parse(demo.dataset.window.history_start)
const end = Date.parse(demo.dataset.cutoff)

export function xOfTime(iso: string): number {
  return LEFT + ((Date.parse(iso) - start) / (end - start)) * (RIGHT - LEFT)
}

const sources = Object.entries(demo.client.source_counts)
const laneOf = new Map(sources.map(([name], index) => [name, index]))

const TICKS = (() => {
  const out: { x: number; text: string }[] = []
  const first = Number(localDay(demo.dataset.window.history_start).slice(0, 4))
  const last = Number(localDay(demo.dataset.cutoff).slice(0, 4))
  for (let year = first; year <= last; year++) {
    for (const month of [0, 3, 6, 9]) {
      const iso = new Date(localMidnight(year, month)).toISOString()
      const x = xOfTime(iso)
      if (x >= LEFT - 0.01 && x <= RIGHT + 0.01) out.push({ x, text: `${year}-${String(month + 1).padStart(2, '0')}` })
    }
  }
  return out
})()

const MARK = new THREE.PlaneGeometry(1, 1)

function Events({ reveal }: { reveal: boolean }) {
  const ref = useRef<THREE.InstancedMesh>(null)
  const progress = useRef(reveal ? 1 : 0)

  const layout = useMemo(
    () =>
      timeline.map((event) => ({
        x: xOfTime(event.t),
        y: LANE_Y0 + (laneOf.get(event.source) ?? 0) * LANE_STEP,
        color: new THREE.Color(sourceColor[event.source] ?? C.muted),
      })),
    [],
  )

  useLayoutEffect(() => {
    const mesh = ref.current
    if (!mesh) return
    layout.forEach((item, index) => mesh.setColorAt(index, item.color))
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [layout])

  useFrame((_, delta) => {
    const mesh = ref.current
    if (!mesh) return

    const target = reveal ? 1 : 0
    const before = progress.current
    progress.current = reveal ? Math.min(1, progress.current + delta / 2.2) : 0

    if (before === progress.current && mesh.userData.ready) return
    mesh.userData.ready = true

    const matrix = new THREE.Matrix4()
    const edge = LEFT + progress.current * (RIGHT - LEFT + 1)

    layout.forEach((item, index) => {
      const scale = target === 0 ? 0 : THREE.MathUtils.clamp((edge - item.x) * 1.5, 0, 1)
      // Невидимое событие — нулевой размер, а не тонкая черта.
      matrix.makeScale(scale === 0 ? 0 : 0.07, 0.46 * scale, 1)
      matrix.setPosition(item.x, item.y, 0)
      mesh.setMatrixAt(index, matrix)
    })

    mesh.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={ref} args={[MARK, undefined, layout.length]} renderOrder={3}>
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  )
}

function Content() {
  const { beat } = useRegionView()

  const heroX = xOfTime(hero.time)
  const heroY = LANE_Y0 + (laneOf.get(hero.source) ?? 0) * LANE_STEP

  const clientAt: Vec3 = [0, 5.2, 0]

  return (
    <group>
      {/* Клиент */}
      <group position={clientAt}>
        <mesh>
          <ringGeometry args={[0.72, 0.8, 48]} />
          <meshBasicMaterial color={C.text} />
        </mesh>
        <mesh>
          <circleGeometry args={[0.32, 32]} />
          <meshBasicMaterial color={C.text2} />
        </mesh>
        <Label size={0.62} position={[1.3, 0.25, 0]} anchorY="middle">
          {'Клиент банка'}
        </Label>
        <Label mono size={0.38} color={C.text2} position={[1.3, -0.45, 0]}>
          {`${demo.client.client_id} · ${int(demo.client.n_events)} событий`}
        </Label>
      </group>

      {/* Ось времени */}
      <Wire points={[[LEFT, AXIS_Y, 0], [RIGHT + 0.6, AXIS_Y, 0]]} accent={C.muted} width={1.2} opacity={0.9} />
      {TICKS.map((tick) => (
        <group key={tick.text}>
          <Wire points={[[tick.x, AXIS_Y, 0], [tick.x, AXIS_Y - 0.18, 0]]} accent={C.muted} width={1} opacity={0.8} />
          <Label mono size={0.3} color={C.muted} anchorX="center" anchorY="top" position={[tick.x, AXIS_Y - 0.3, 0]}>
            {tick.text}
          </Label>
        </group>
      ))}

      {/* События */}
      <Events reveal={beat >= 1} />

      <Appear show={beat >= 1} delay={0.4}>
        {sources.map(([name, count], index) => (
          <Label key={name} mono size={0.3} color={sourceColor[name] ?? C.muted} anchorX="right" position={[LEFT - 0.35, LANE_Y0 + index * LANE_STEP, 0]}>
            {`${name} ${count}`}
          </Label>
        ))}
      </Appear>

      {/* Сквозное событие */}
      <Appear show={beat >= 1} delay={2.2} position={[heroX, heroY, 0.05]}>
        <mesh>
          <ringGeometry args={[0.26, 0.34, 32]} />
          <meshBasicMaterial color={C.cyan} />
        </mesh>
        <Wire points={[[0, 0.35, 0], [0, 1.6, 0]]} accent={C.cyan} width={1.2} opacity={0.8} />
        <Label mono size={0.36} color={C.cyan} anchorX="center" anchorY="bottom" position={[0, 1.75, 0]}>
          {`${hero.type} · ${String(hero.raw.merchant_name)}`}
        </Label>
      </Appear>

      {/* cutoff T */}
      <Appear show={beat >= 2} position={[RIGHT + 0.6, 0, 0]}>
        <Wire points={[[0, AXIS_Y, 0], [0, 3.4, 0]]} accent={C.violet} width={1.6} opacity={0.9} />
        <Label size={0.5} color={C.violet} anchorX="center" anchorY="bottom" position={[0, 3.6, 0]}>
          {'cutoff T'}
        </Label>
        <Label mono size={0.32} color={C.text2} anchorX="center" anchorY="top" position={[0, AXIS_Y - 0.75, 0]}>
          {localDay(demo.dataset.cutoff)}
        </Label>
      </Appear>
    </group>
  )
}

export function ClientRegion() {
  return (
    <Region id="client" size={[34, 16]}>
      <Content />
    </Region>
  )
}
