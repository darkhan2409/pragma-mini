import { CameraControls, CameraControlsImpl, PerformanceMonitor } from '@react-three/drei'
import { Canvas, useThree } from '@react-three/fiber'
import { Suspense, useEffect, useRef, useState } from 'react'
import { ANCHOR, RAIL_BRANCHES, RAIL_MAIN, REGION_TITLE, type RegionId, type Vec3 } from '../content/layout'
import { PARTS } from '../content/parts'
import { FOV, SHOTS, TAN, type Shot, type ShotId } from '../content/shots'
import { STEPS, shotOf } from '../content/steps'
import { useStore } from '../store'
import { color as C } from '../theme'
import { Label, Wire } from './primitives'
import { BackpropRegion } from './regions/Backprop'
import { CalendarRegion } from './regions/Calendar'
import { ClientRegion } from './regions/Client'
import { EmbeddingRegion } from './regions/Embedding'
import { EventRegion } from './regions/Event'
import { EventEncoderRegion } from './regions/EventEncoder'
import { ClientEmbeddingRegion, DownstreamRegion } from './regions/Final'
import { HistoryRegion } from './regions/History'
import { HistoryEncoderRegion } from './regions/HistoryEncoder'
import { LoopRegion } from './regions/Loop'
import { MlmRegion } from './regions/Mlm'
import { ProfileRegion } from './regions/Profile'
import { RopeRegion } from './regions/Rope'
import { TokensRegion } from './regions/Tokens'

// ============================================================
// СЦЕНА
// ============================================================
//
// Один Canvas на всё время показа. Камера — CameraControls:
// в PRESENT ввод выключен и камера летит по кадрам шагов, в
// EXPLORE управляет мышь, а клик по региону подводит к нему.
// Пока открыт 2D-экран, кадры не рисуются.
// ============================================================

const OFF = CameraControlsImpl.ACTION.NONE

// Регион EXPLORE → кадр.
const PART_SHOT: Record<string, ShotId> = {
  client: 'client',
  event: 'event',
  tokens: 'tokens',
  embedding: 'embedding',
  eventEncoder: 'eventEncoder',
  calendar: 'calendar',
  profile: 'profile',
  history: 'history',
  rope: 'rope',
  historyEncoder: 'historyEncoder',
  mlm: 'mlm',
  loop: 'loop',
  clientEmbedding: 'final',
  downstream: 'downstream',
}

// Панель деталей EXPLORE закрывает справа столько пикселей (с
// отступами): кадр детали отъезжает и сдвигается, чтобы регион
// остался в свободной части экрана.
const DRAWER = 464

function beside(shot: Shot, width: number, height: number): Shot {
  const free = Math.max(0.5, (width - DRAWER) / width)
  const [px, py, pz] = shot.position
  const [tx, ty, tz] = shot.target
  const distance = (pz - tz) / free
  const shift = (1 - free) * distance * TAN * (width / height)
  return { position: [px + shift, py, tz + distance], target: [tx + shift, ty, tz] }
}

function look(controls: CameraControlsImpl, shot: Shot, smooth: boolean) {
  const [px, py, pz] = shot.position
  const [tx, ty, tz] = shot.target
  void controls.setLookAt(px, py, pz, tx, ty, tz, smooth)
}

function CameraRig() {
  const ref = useRef<CameraControlsImpl>(null)
  const mode = useStore((s) => s.mode)
  const step = useStore((s) => s.step)
  const beat = useStore((s) => s.beat)
  const selected = useStore((s) => s.selected)
  const recenter = useStore((s) => s.recenter)
  const width = useThree((s) => s.size.width)
  const height = useThree((s) => s.size.height)
  const first = useRef(true)

  // Ввод: в PRESENT камера только по сценарию.
  useEffect(() => {
    const controls = ref.current
    if (!controls) return
    const explore = mode === 'explore'
    controls.mouseButtons.left = explore ? CameraControlsImpl.ACTION.ROTATE : OFF
    controls.mouseButtons.right = explore ? CameraControlsImpl.ACTION.TRUCK : OFF
    controls.mouseButtons.middle = explore ? CameraControlsImpl.ACTION.DOLLY : OFF
    controls.mouseButtons.wheel = explore ? CameraControlsImpl.ACTION.DOLLY : OFF
    controls.touches.one = explore ? CameraControlsImpl.ACTION.TOUCH_ROTATE : OFF
    controls.touches.two = explore ? CameraControlsImpl.ACTION.TOUCH_DOLLY_TRUCK : OFF
    controls.touches.three = OFF
    controls.dollyToCursor = explore
    controls.smoothTime = explore ? 0.3 : 0.85
    controls.minDistance = 4
    controls.maxDistance = 320
  }, [mode])

  // Кадр шага (PRESENT) или выбранной детали (EXPLORE).
  useEffect(() => {
    const controls = ref.current
    if (!controls) return

    let shot: Shot | undefined

    if (mode === 'present') {
      const id = shotOf(STEPS[step], beat)
      shot = id ? SHOTS[id] : undefined
    } else if (selected && PART_SHOT[selected]) {
      shot = beside(SHOTS[PART_SHOT[selected]], width, height)
    }

    if (shot) {
      look(controls, shot, !first.current)
      first.current = false
    }
  }, [mode, step, beat, selected, width, height])

  // Общий вид EXPLORE: при входе в режим и по клавише R.
  useEffect(() => {
    const controls = ref.current
    if (!controls || mode !== 'explore') return
    look(controls, SHOTS.overview, !first.current)
    first.current = false
  }, [mode, recenter])

  return <CameraControls ref={ref} makeDefault />
}

// Рельс: пол под регионами, связывающий их по пути данных. Виден
// только на общих планах и в EXPLORE; заголовки — у регионов шага.
function Rail() {
  const step = useStore((s) => s.step)
  const beat = useStore((s) => s.beat)
  const mode = useStore((s) => s.mode)
  const selected = useStore((s) => s.selected)
  const id = STEPS[step].id
  const explore = mode === 'explore'
  const overview = explore || id === 'backprop' || (id === 'final' && beat === 0)
  // Крупные заголовки — для общего плана; вблизи выбранной детали мешают.
  const titles = overview && !(explore && selected !== null)
  const noTraining = id === 'final' || id === 'downstream'
  const shown = (region: RegionId) => explore || (STEPS[step].regions[region] ?? 'hidden') !== 'hidden'

  if (!overview) return null

  const floor = (region: RegionId): Vec3 => {
    const [x, y] = ANCHOR[region]
    return [x, y - 8.2, -3]
  }

  const main = RAIL_MAIN.map(floor)

  return (
    <group>
      <Wire points={main} accent={C.line} width={1.4} opacity={0.9} />
      {RAIL_BRANCHES.filter(([a, b]) => !(noTraining && (a === 'mlm' || b === 'mlm' || b === 'loop'))).map(([a, b]) => (
        <Wire key={`${a}-${b}`} points={[floor(a), floor(b)]} accent={C.line} width={1.2} opacity={0.7} dashed />
      ))}
      {titles
        ? (Object.keys(REGION_TITLE) as RegionId[])
            .filter((region) => region !== 'clientEmbedding' && shown(region))
            .filter((region) => !(id === 'backprop' && region === 'mlm'))
            .filter((region) => !(noTraining && (region === 'mlm' || region === 'loop')))
            .map((region) => {
              const [x, y] = ANCHOR[region]
              return (
                <Label key={region} size={2.0} anchorX="center" color={C.text2} position={[x, y + 9.4, 0]}>
                  {REGION_TITLE[region] ?? region}
                </Label>
              )
            })
        : null}
    </group>
  )
}

export function Stage() {
  const screen = useStore((s) => (s.mode === 'present' ? STEPS[s.step].screen : undefined))
  const lowPower = useStore((s) => s.lowPower)
  const mode = useStore((s) => s.mode)
  const next = useStore((s) => s.next)
  const select = useStore((s) => s.select)
  const [dpr, setDpr] = useState(1.5)

  return (
    <div
      className="canvas-wrap"
      style={{ visibility: screen ? 'hidden' : 'visible' }}
      onClick={() => {
        if (mode === 'present') next()
      }}
    >
      <Canvas
        dpr={lowPower ? 1 : [1, dpr]}
        frameloop={screen ? 'never' : 'always'}
        camera={{ fov: FOV, near: 0.1, far: 900, position: [0, 3, 40] }}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        onPointerMissed={() => {
          if (mode === 'explore') select(null)
        }}
      >
        <color attach="background" args={[C.bg]} />
        <fog attach="fog" args={[C.bg, 160, 520]} />
        <PerformanceMonitor onDecline={() => setDpr(1)} onIncline={() => setDpr(1.5)} />
        <Suspense fallback={null}>
          <CameraRig />
          <Rail />
          <ClientRegion />
          <EventRegion />
          <TokensRegion />
          <EmbeddingRegion />
          <EventEncoderRegion />
          <CalendarRegion />
          <ProfileRegion />
          <HistoryRegion />
          <RopeRegion />
          <HistoryEncoderRegion />
          <MlmRegion />
          <BackpropRegion />
          <LoopRegion />
          <ClientEmbeddingRegion />
          <DownstreamRegion />
        </Suspense>
      </Canvas>
    </div>
  )
}

// Для EXPLORE: известна ли деталь.
export function hasPart(id: string | null): id is string {
  return id !== null && id in PARTS
}
