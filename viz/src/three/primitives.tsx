import { Line, Text } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { createContext, useContext, useLayoutEffect, useMemo, useRef, type ReactNode, type RefObject } from 'react'
import * as THREE from 'three'
import type { Line2, LineSegments2 } from 'three-stdlib'
import { color as C, FONT } from '../theme'
import type { Vec3 } from '../content/layout'

// ============================================================
// ПРИМИТИВЫ СЦЕНЫ
// ============================================================
//
// Материалы неосвещаемые, теней и HDR нет. Появление элементов —
// масштабом, а прозрачность всего региона ведёт useRegionFade:
// поэлементная прозрачность с ней не спорит.
// ============================================================

// ------------------------------------------------------------
// Затухание региона
// ------------------------------------------------------------

function applyOpacity(root: THREE.Object3D, value: number) {
  root.traverse((object) => {
    const target = object as THREE.Mesh & { fillOpacity?: number; outlineOpacity?: number }
    const material = target.material as THREE.Material | THREE.Material[] | undefined

    if (material) {
      for (const item of Array.isArray(material) ? material : [material]) {
        if (item.userData.base === undefined) item.userData.base = item.opacity ?? 1
        item.transparent = true
        item.opacity = item.userData.base * value
      }
    }
  })
}

export function useRegionFade(ref: RefObject<THREE.Group | null>, target: number) {
  const current = useRef(0)

  useFrame((_, delta) => {
    const group = ref.current
    if (!group) return

    const next = THREE.MathUtils.damp(current.current, target, 5, Math.min(delta, 0.1))
    const settled = Math.abs(next - target) < 0.002 ? target : next

    if (settled === current.current && group.userData.faded) return

    current.current = settled
    group.userData.faded = true
    group.visible = settled > 0.004

    if (group.visible) applyOpacity(group, settled)
  })
}

// ------------------------------------------------------------
// Появление по масштабу
// ------------------------------------------------------------

export function Appear({
  show,
  delay = 0,
  children,
  position,
  speed = 7,
}: {
  show: boolean
  delay?: number
  children: ReactNode
  position?: Vec3
  speed?: number
}) {
  const ref = useRef<THREE.Group>(null)
  const since = useRef<number | null>(null)
  const scale = useRef(show ? 1 : 0)

  useFrame((state, delta) => {
    const group = ref.current
    if (!group) return

    if (show && since.current === null) since.current = state.clock.elapsedTime
    if (!show) since.current = null

    const ready = show && since.current !== null && state.clock.elapsedTime - since.current >= delay
    const target = ready ? 1 : 0

    scale.current = THREE.MathUtils.damp(scale.current, target, speed, Math.min(delta, 0.1))
    if (Math.abs(scale.current - target) < 0.002) scale.current = target

    group.scale.setScalar(Math.max(scale.current, 1e-4))
    group.visible = scale.current > 0.01
  })

  return (
    <group ref={ref} position={position} scale={show ? 1 : 1e-4}>
      {children}
    </group>
  )
}

// ------------------------------------------------------------
// Текст
// ------------------------------------------------------------

type Anchor = 'left' | 'center' | 'right'
type AnchorY = 'top' | 'middle' | 'bottom'

export function Label({
  children,
  size = 0.55,
  color = C.text,
  mono = false,
  anchorX = 'left',
  anchorY = 'middle',
  position,
  maxWidth,
  lineHeight,
}: {
  children: string
  size?: number
  color?: string
  mono?: boolean
  anchorX?: Anchor
  anchorY?: AnchorY
  position?: Vec3
  maxWidth?: number
  lineHeight?: number
}) {
  return (
    <Text
      font={mono ? FONT.mono : FONT.sans}
      fontSize={size}
      color={color}
      anchorX={anchorX}
      anchorY={anchorY}
      position={position}
      maxWidth={maxWidth}
      lineHeight={lineHeight}
      renderOrder={10}
    >
      {children}
    </Text>
  )
}

// ------------------------------------------------------------
// Скруглённый прямоугольник: заливка и контур
// ------------------------------------------------------------

function roundedPoints(width: number, height: number, radius: number, segments = 6): THREE.Vector2[] {
  const r = Math.min(radius, width / 2, height / 2)
  const w = width / 2
  const h = height / 2
  const points: THREE.Vector2[] = []
  const corners: [number, number, number][] = [
    [w - r, h - r, 0],
    [-w + r, h - r, Math.PI / 2],
    [-w + r, -h + r, Math.PI],
    [w - r, -h + r, (3 * Math.PI) / 2],
  ]
  for (const [cx, cy, start] of corners) {
    for (let i = 0; i <= segments; i++) {
      const a = start + (i / segments) * (Math.PI / 2)
      points.push(new THREE.Vector2(cx + r * Math.cos(a), cy + r * Math.sin(a)))
    }
  }
  return points
}

export function Card({
  width,
  height,
  radius = 0.28,
  fill = C.surface,
  fillOpacity = 0.92,
  stroke = C.line,
  strokeOpacity = 1,
  lineWidth = 1.2,
  position,
  children,
}: {
  width: number
  height: number
  radius?: number
  fill?: string
  fillOpacity?: number
  stroke?: string
  strokeOpacity?: number
  lineWidth?: number
  position?: Vec3
  children?: ReactNode
}) {
  const { shape, outline } = useMemo(() => {
    const points = roundedPoints(width, height, radius)
    const shape = new THREE.Shape(points)
    const outline = [...points, points[0]].map((p) => [p.x, p.y, 0.001] as Vec3)
    return { shape, outline }
  }, [width, height, radius])

  return (
    <group position={position}>
      <mesh renderOrder={1}>
        <shapeGeometry args={[shape]} />
        <meshBasicMaterial color={fill} transparent opacity={fillOpacity} depthWrite={false} />
      </mesh>
      <Line points={outline} color={stroke} lineWidth={lineWidth} transparent opacity={strokeOpacity} renderOrder={2} />
      {children}
    </group>
  )
}

// Ширина фишки под моноширинный текст.
export function chipWidth(text: string, size: number): number {
  return Math.max(1.2, text.length * size * 0.6 + 0.7)
}

// Фишка токена: скруглённая рамка и текст моноширинным.
export function Chip({
  text,
  position,
  accent = C.cyan,
  width,
  size = 0.42,
  strong = false,
  textColor,
}: {
  text: string
  position?: Vec3
  accent?: string
  width?: number
  size?: number
  strong?: boolean
  textColor?: string
}) {
  const w = width ?? chipWidth(text, size)
  const h = size * 2.1
  return (
    <Card
      width={w}
      height={h}
      radius={0.18}
      position={position}
      fill={strong ? C.surface2 : C.surface}
      stroke={accent}
      strokeOpacity={strong ? 0.95 : 0.55}
    >
      <Label mono size={size} anchorX="center" color={textColor ?? (strong ? C.text : C.text2)} position={[0, 0, 0.02]}>
        {text}
      </Label>
    </Card>
  )
}

// ------------------------------------------------------------
// Вектор [128]: столбец из 16 ячеек
// ------------------------------------------------------------

function seeded(seed: number) {
  let s = (seed * 2654435761) >>> 0 || 1
  return () => {
    s ^= s << 13
    s ^= s >>> 17
    s ^= s << 5
    return ((s >>> 0) % 10000) / 10000
  }
}

const CELL = new THREE.PlaneGeometry(1, 1)

export function Glyph({
  seed,
  accent = C.cyan,
  position,
  height = 3.2,
  width = 0.42,
  caption = '[128]',
  cells = 16,
  label,
}: {
  seed: number
  accent?: string
  position?: Vec3
  height?: number
  width?: number
  caption?: string | null
  cells?: number
  label?: string
}) {
  const ref = useRef<THREE.InstancedMesh>(null)

  useLayoutEffect(() => {
    const mesh = ref.current
    if (!mesh) return
    const random = seeded(seed + 17)
    const base = new THREE.Color(C.surface2)
    const bright = new THREE.Color(accent)
    const matrix = new THREE.Matrix4()
    const tint = new THREE.Color()
    const step = height / cells
    for (let i = 0; i < cells; i++) {
      matrix.makeScale(width, step * 0.8, 1)
      matrix.setPosition(0, height / 2 - step * (i + 0.5), 0)
      mesh.setMatrixAt(i, matrix)
      tint.copy(base).lerp(bright, 0.18 + 0.82 * random())
      mesh.setColorAt(i, tint)
    }
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [seed, accent, height, width, cells])

  return (
    <group position={position}>
      <instancedMesh ref={ref} args={[CELL, undefined, cells]} renderOrder={3}>
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      <Line
        points={[
          [-width / 2 - 0.12, height / 2 + 0.12, 0],
          [width / 2 + 0.12, height / 2 + 0.12, 0],
          [width / 2 + 0.12, -height / 2 - 0.12, 0],
          [-width / 2 - 0.12, -height / 2 - 0.12, 0],
          [-width / 2 - 0.12, height / 2 + 0.12, 0],
        ]}
        color={accent}
        lineWidth={1}
        transparent
        opacity={0.5}
      />
      {caption ? (
        <Label mono size={0.34} anchorX="center" anchorY="top" color={C.text2} position={[0, -height / 2 - 0.35, 0]}>
          {caption}
        </Label>
      ) : null}
      {label ? (
        <Label size={0.42} anchorX="center" anchorY="bottom" color={accent} position={[0, height / 2 + 0.4, 0]}>
          {label}
        </Label>
      ) : null}
    </group>
  )
}

// ------------------------------------------------------------
// Линии
// ------------------------------------------------------------

export function Wire({
  points,
  accent = C.cyan,
  width = 1.4,
  opacity = 0.7,
  dashed = false,
  flow = 0,
}: {
  points: Vec3[]
  accent?: string
  width?: number
  opacity?: number
  dashed?: boolean
  flow?: number
}) {
  const ref = useRef<Line2 | LineSegments2>(null)

  useFrame((_, delta) => {
    if (!flow || !ref.current) return
    const material = ref.current.material as THREE.Material & { dashOffset: number }
    material.dashOffset -= flow * delta
  })

  return (
    <Line
      ref={ref}
      points={points}
      color={accent}
      lineWidth={width}
      transparent
      opacity={opacity}
      dashed={dashed || flow !== 0}
      dashSize={0.5}
      gapSize={0.35}
      renderOrder={2}
    />
  )
}

// Дуга между двумя точками: для внимания и связей.
export function Arc({
  from,
  to,
  lift = 1.5,
  accent = C.cyan,
  width = 1.2,
  opacity = 0.6,
  dashed = false,
  flow = 0,
}: {
  from: Vec3
  to: Vec3
  lift?: number
  accent?: string
  width?: number
  opacity?: number
  dashed?: boolean
  flow?: number
}) {
  const points = useMemo(() => {
    const a = new THREE.Vector3(...from)
    const b = new THREE.Vector3(...to)
    const mid = a.clone().add(b).multiplyScalar(0.5)
    mid.y += lift
    return new THREE.QuadraticBezierCurve3(a, mid, b).getPoints(28).map((p) => [p.x, p.y, p.z] as Vec3)
  }, [from, to, lift])

  return <Wire points={points} accent={accent} width={width} opacity={opacity} dashed={dashed} flow={flow} />
}

// Стрелка-указатель: линия и наконечник.
export function Arrow({ from, to, accent = C.muted, width = 1.2, opacity = 0.8 }: { from: Vec3; to: Vec3; accent?: string; width?: number; opacity?: number }) {
  const head = useMemo(() => {
    const a = new THREE.Vector3(...from)
    const b = new THREE.Vector3(...to)
    const dir = b.clone().sub(a).normalize()
    const side = new THREE.Vector3(-dir.y, dir.x, 0).multiplyScalar(0.18)
    const back = b.clone().sub(dir.clone().multiplyScalar(0.32))
    return [
      [back.x + side.x, back.y + side.y, back.z] as Vec3,
      [b.x, b.y, b.z] as Vec3,
      [back.x - side.x, back.y - side.y, back.z] as Vec3,
    ]
  }, [from, to])

  return (
    <group>
      <Wire points={[from, to]} accent={accent} width={width} opacity={opacity} />
      <Wire points={head} accent={accent} width={width} opacity={opacity} />
    </group>
  )
}

// ------------------------------------------------------------
// Блок: полупрозрачная рамка
// ------------------------------------------------------------

export function Block({
  size,
  position,
  accent = C.cyan,
  faceOpacity = 0.045,
  edgeOpacity = 0.45,
}: {
  size: Vec3
  position?: Vec3
  accent?: string
  faceOpacity?: number
  edgeOpacity?: number
}) {
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(...size)), [size])

  return (
    <group position={position}>
      <mesh renderOrder={0}>
        <boxGeometry args={size} />
        <meshBasicMaterial color={accent} transparent opacity={faceOpacity} depthWrite={false} />
      </mesh>
      <lineSegments geometry={edges} renderOrder={1}>
        <lineBasicMaterial color={accent} transparent opacity={edgeOpacity} />
      </lineSegments>
    </group>
  )
}

// ------------------------------------------------------------
// Окружность с указателем: календарь и TimeRoPE
// ------------------------------------------------------------

export function Dial({
  radius = 1.2,
  angle,
  accent = C.violet,
  position,
  ghost,
  label,
}: {
  radius?: number
  angle: number
  accent?: string
  position?: Vec3
  ghost?: number
  label?: string
}) {
  const circle = useMemo(
    () =>
      Array.from({ length: 65 }, (_, i) => {
        const a = (i / 64) * Math.PI * 2
        return [radius * Math.cos(a), radius * Math.sin(a), 0] as Vec3
      }),
    [radius],
  )

  const tip = (a: number): Vec3 => [radius * Math.cos(a), radius * Math.sin(a), 0.01]

  return (
    <group position={position}>
      <Wire points={circle} accent={C.line} width={1.2} opacity={1} />
      {ghost !== undefined ? <Wire points={[[0, 0, 0], tip(ghost)]} accent={C.muted} width={1.2} opacity={0.7} dashed /> : null}
      <Wire points={[[0, 0, 0.01], tip(angle)]} accent={accent} width={2.2} opacity={0.95} />
      <mesh position={tip(angle)}>
        <circleGeometry args={[0.1, 16]} />
        <meshBasicMaterial color={accent} />
      </mesh>
      {label ? (
        <Label size={0.36} anchorX="center" anchorY="top" color={C.text2} position={[0, -radius - 0.3, 0]}>
          {label}
        </Label>
      ) : null}
    </group>
  )
}

// ------------------------------------------------------------
// Контекст региона: состояние и бит
// ------------------------------------------------------------

export interface RegionView {
  state: 'focus' | 'context' | 'hidden'
  beat: number
  explore: boolean
}

export const RegionContext = createContext<RegionView>({ state: 'hidden', beat: 0, explore: false })

export function useRegionView(): RegionView {
  return useContext(RegionContext)
}
