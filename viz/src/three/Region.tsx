import { useRef, type ReactNode } from 'react'
import * as THREE from 'three'
import { ANCHOR, type RegionId } from '../content/layout'
import { STEPS } from '../content/steps'
import { beatFor, useStore } from '../store'
import { FADE } from '../theme'
import { RegionContext, useRegionFade } from './primitives'

// ============================================================
// РЕГИОН СЦЕНЫ
// ============================================================
//
// Регион монтируется один раз и дальше только затухает или
// проявляется: состояние берётся из шага (focus / context /
// hidden). Бит — подшаг его собственного шага; в контексте регион
// показан целиком. В EXPLORE видно всё, клик по зоне региона
// открывает его детали, а соседи выбранного уходят в контекст.
// ============================================================

// Свой шаг региона: по нему считается бит.
const OWN_STEP: Record<RegionId, string> = {
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
  backprop: 'backprop',
  loop: 'loop',
  clientEmbedding: 'final',
  downstream: 'downstream',
}

// Детали EXPLORE по региону.
const PART: Partial<Record<RegionId, string>> = {
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
  clientEmbedding: 'clientEmbedding',
  downstream: 'downstream',
}

export function Region({
  id,
  children,
  size = [30, 14],
  offset = [0, 0],
}: {
  id: RegionId
  children: ReactNode
  size?: [number, number]
  offset?: [number, number]
}) {
  const mode = useStore((s) => s.mode)
  const step = useStore((s) => s.step)
  const beat = useStore((s) => beatFor(OWN_STEP[id], s))
  const select = useStore((s) => s.select)
  const selected = useStore((s) => s.selected)

  const ref = useRef<THREE.Group>(null)

  const explore = mode === 'explore'
  const own = STEPS.find((item) => item.id === OWN_STEP[id])
  const lastBeat = own ? own.beats.length - 1 : 0

  const part = PART[id]

  const state = explore
    ? id === 'backprop'
      ? 'hidden'
      : selected !== null && selected !== part
        ? 'context'
        : 'focus'
    : (STEPS[step].regions[id] ?? 'hidden')
  const shown = explore ? lastBeat : state === 'context' ? lastBeat : Math.max(0, beat)

  useRegionFade(ref, FADE[state])

  return (
    <RegionContext.Provider value={{ state, beat: shown, explore }}>
      <group ref={ref} position={ANCHOR[id]}>
        {children}
        {explore && part ? (
          <mesh
            position={[offset[0], offset[1], -0.5]}
            onClick={(event) => {
              event.stopPropagation()
              select(part)
            }}
            onPointerOver={(event) => {
              event.stopPropagation()
              document.body.style.cursor = 'pointer'
            }}
            onPointerOut={() => {
              document.body.style.cursor = ''
            }}
          >
            <planeGeometry args={size} />
            <meshBasicMaterial transparent opacity={selected === part ? 0.035 : 0} depthWrite={false} color="#ffffff" />
          </mesh>
        ) : null}
      </group>
    </RegionContext.Provider>
  )
}
