import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { demo, hero, localDay } from '../../data/demo'
import { color as C } from '../../theme'
import { Appear, Glyph, Label, Wire, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 8. ИСТОРИЯ: [USR] И ДАТИРОВАННЫЕ СОБЫТИЯ ДО T
// ============================================================
//
// Бит 0 — события по времени, [USR] в голове последовательности.
// Бит 1 — ось становится позицией TimeRoPE: давность до T,
// 8·log1p(Δt/8). [USR] уезжает в позицию 0, к самому T.
// ============================================================

const LEFT = -12.5
const RIGHT = 11.5
const Y = 0.2

const timeline = demo.client.timeline
const start = Date.parse(demo.dataset.window.history_start)
const end = Date.parse(demo.dataset.cutoff)
const maxLog = Math.max(...timeline.map((event) => event.time_log))

export function xByTime(iso: string): number {
  return LEFT + ((Date.parse(iso) - start) / (end - start)) * (RIGHT - LEFT)
}

export function xByLog(position: number): number {
  return RIGHT - (position / maxLog) * (RIGHT - LEFT)
}

const BAR = new THREE.PlaneGeometry(1, 1)

function Bars({ log }: { log: boolean }) {
  const ref = useRef<THREE.InstancedMesh>(null)
  const mix = useRef(log ? 1 : 0)

  const points = useMemo(
    () =>
      timeline.map((event, index) => ({
        a: xByTime(event.t),
        b: xByLog(event.time_log),
        hero: index === hero.index,
        height: 0.6 + Math.min(1.6, event.n_tokens / 14),
      })),
    [],
  )

  useLayoutEffect(() => {
    const mesh = ref.current
    if (!mesh) return
    const dim = new THREE.Color(C.cyan).lerp(new THREE.Color(C.bg), 0.55)
    const bright = new THREE.Color(C.cyan)
    points.forEach((point, index) => mesh.setColorAt(index, point.hero ? bright : dim))
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [points])

  useFrame((_, delta) => {
    const mesh = ref.current
    if (!mesh) return
    const before = mix.current
    mix.current = THREE.MathUtils.damp(mix.current, log ? 1 : 0, 3, Math.min(delta, 0.1))
    if (Math.abs(mix.current - (log ? 1 : 0)) < 0.001) mix.current = log ? 1 : 0
    if (before === mix.current && mesh.userData.ready) return
    mesh.userData.ready = true
    const matrix = new THREE.Matrix4()
    points.forEach((point, index) => {
      const x = point.a + (point.b - point.a) * mix.current
      matrix.makeScale(point.hero ? 0.16 : 0.06, point.hero ? 2.6 : point.height, 1)
      matrix.setPosition(x, Y + (point.hero ? 1.3 : point.height / 2), point.hero ? 0.05 : 0)
      mesh.setMatrixAt(index, matrix)
    })
    mesh.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={ref} args={[BAR, undefined, points.length]} renderOrder={3}>
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  )
}

function UsrSlot({ log }: { log: boolean }) {
  const ref = useRef<THREE.Group>(null)
  const x = useRef(log ? RIGHT + 1.6 : LEFT - 1.8)

  useFrame((_, delta) => {
    if (!ref.current) return
    x.current = THREE.MathUtils.damp(x.current, log ? RIGHT + 1.6 : LEFT - 1.8, 3, Math.min(delta, 0.1))
    ref.current.position.x = x.current
  })

  return (
    <group ref={ref} position={[x.current, Y + 1.4, 0]}>
      <Glyph seed={4} accent={C.amber} height={2.8} width={0.4} caption={null} label="[USR]" />
      <Label mono size={0.3} anchorX="center" color={C.text2} position={[0, -1.9, 0]}>
        {log ? 'позиция 0' : 'вектор анкеты'}
      </Label>
    </group>
  )
}

function Content() {
  const { beat } = useRegionView()
  const log = beat >= 1

  const heroX = log ? xByLog(hero.time_log) : xByTime(hero.time)

  return (
    <group>
      <Wire points={[[LEFT - 0.4, Y, 0], [RIGHT + 0.4, Y, 0]]} accent={C.muted} width={1.2} opacity={0.9} />
      <Bars log={log} />
      <UsrSlot log={log} />

      <group position={[RIGHT + 0.4, 0, 0]}>
        <Wire points={[[0, Y - 0.6, 0], [0, Y + 3.8, 0]]} accent={C.violet} width={1.6} opacity={0.9} />
        <Label size={0.44} anchorX="center" color={C.violet} position={[0, Y + 4.3, 0]}>
          {'T'}
        </Label>
      </group>

      <Label mono size={0.34} color={C.cyan} anchorX="center" position={[heroX, Y + 3.3, 0]}>
        {`${hero.type} · ${String(hero.raw.merchant_name)}`}
      </Label>

      <Label size={0.38} color={C.text2} position={[LEFT - 0.4, Y - 1.0, 0]}>
        {log
          ? 'ось — позиция TimeRoPE: давность до T = 8·log1p(Δt/8), Δt в секундах'
          : `ось — время: ${localDay(demo.dataset.window.history_start)} → ${localDay(demo.dataset.cutoff)}`}
      </Label>

      <Appear show position={[LEFT - 0.4, 5.6, 0]}>
        <Label size={0.42} color={C.text}>
          {`[USR] + ${demo.client.n_events} датированных векторов событий`}
        </Label>
        <Label size={0.32} color={C.text2} position={[0, -0.6, 0]}>
          {'от старых к новым · анкета и события встречаются здесь впервые'}
        </Label>
      </Appear>
    </group>
  )
}

export function HistoryRegion() {
  return (
    <Region id="history" size={[32, 12]}>
      <Content />
    </Region>
  )
}
