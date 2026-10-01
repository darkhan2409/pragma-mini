import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { demo, int, num } from '../../data/demo'
import type { Vec3 } from '../../content/layout'
import { useStore } from '../../store'
import { color as C } from '../../theme'
import { Appear, Chip, Label, Wire, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 13. ЦИКЛ ОБУЧЕНИЯ И РАСПИСАНИЕ LR
// ============================================================
//
// Кривая LR — та же формула, что в src/mlm/train.py (lr_factor):
// линейный разгон warmup_steps шагов, затем cosine до
// min_learning_rate к концу горизонта. Горизонт и val loss по
// эпохам — из прогона.
// ============================================================

const STATIONS = ['micro-batch', 'forward', 'loss', 'backward', 'шаг AdamW']
const RING = 4.2
const CENTER: Vec3 = [-8.6, 0.6, 0]

export function lrAt(done: number, warmup: number, total: number, peak: number, floor: number): number {
  if (done < warmup) return (peak * (done + 1)) / warmup
  const p = Math.min(1, (done - warmup) / Math.max(1, total - warmup))
  const factor = floor / peak + (1 - floor / peak) * 0.5 * (1 + Math.cos(Math.PI * p))
  return peak * factor
}

function Pulse() {
  const ref = useRef<THREE.Mesh>(null)
  const lowPower = useStore((s) => s.lowPower)
  useFrame((state) => {
    if (!ref.current || lowPower) return
    const a = Math.PI / 2 - state.clock.elapsedTime * 1.1
    ref.current.position.set(CENTER[0] + RING * Math.cos(a), CENTER[1] + RING * Math.sin(a), 0.05)
  })
  return (
    <mesh ref={ref} position={[CENTER[0], CENTER[1] + RING, 0.05]}>
      <circleGeometry args={[0.2, 20]} />
      <meshBasicMaterial color={C.cyan} />
    </mesh>
  )
}

function Content() {
  const { beat } = useRegionView()
  const t = demo.training
  const run = demo.run
  const perEpoch = demo.batching.micro_batches
  const epochs = Number(run?.plan?.epochs ?? 5)
  const total = Number(run?.total_steps ?? perEpoch * epochs)

  const circle = useMemo(
    () =>
      Array.from({ length: 97 }, (_, i) => {
        const a = (i / 96) * Math.PI * 2
        return [CENTER[0] + RING * Math.cos(a), CENTER[1] + RING * Math.sin(a), 0] as Vec3
      }),
    [],
  )

  // Кривая LR: x — шаг, y — LR.
  const chart = useMemo(() => {
    const x0 = 0.6
    const x1 = 14.4
    const y0 = -3.4
    const y1 = 2.2
    const peak = Number(t.learning_rate)
    const floor = Number(t.min_learning_rate)
    const warmup = Number(t.warmup_steps)
    const points: Vec3[] = []
    for (let i = 0; i <= 240; i++) {
      const done = Math.round((i / 240) * total)
      const lr = lrAt(done, warmup, total, peak, floor)
      points.push([x0 + (i / 240) * (x1 - x0), y0 + (lr / peak) * (y1 - y0), 0])
    }
    const ticks = Array.from({ length: epochs }, (_, i) => ({
      x: x0 + (((i + 1) * perEpoch) / total) * (x1 - x0),
      epoch: i + 1,
      val: run?.epochs.find((item) => item.epoch === i + 1)?.val_loss ?? null,
    }))
    return { x0, x1, y0, y1, points, ticks, peak, floor, warmup }
  }, [t, total, epochs, perEpoch, run])

  return (
    <group>
      <Wire points={circle} accent={C.line} width={1.6} opacity={1} />
      {STATIONS.map((name, index) => {
        const a = Math.PI / 2 - (index / STATIONS.length) * Math.PI * 2
        return <Chip key={name} text={name} strong position={[CENTER[0] + RING * Math.cos(a), CENTER[1] + RING * Math.sin(a), 0.1]} size={0.36} accent={index === 2 || index === 3 ? C.rose : C.cyan} />
      })}
      <Pulse />
      <Label size={0.36} anchorX="center" color={C.text2} position={[CENTER[0], CENTER[1] + 0.35, 0]}>
        {`micro-batch ≈ ${int(demo.batching.mean_tokens)} токенов`}
      </Label>
      <Label size={0.32} anchorX="center" color={C.muted} position={[CENTER[0], CENTER[1] - 0.3, 0]}>
        {`бюджет ${int(Number(t.token_budget))} · без паддинга`}
      </Label>

      <Label size={0.4} color={C.text} position={[-14.6, 6.0, 0]}>
        {`${int(perEpoch)} micro-batch = ${int(perEpoch)} шагов = 1 эпоха · grad_accum_steps ${t.grad_accum_steps}`}
      </Label>

      <Appear show={beat >= 1}>
        <Wire points={chart.points} accent={C.cyan} width={2} opacity={0.95} />
        <Wire points={[[chart.x0, chart.y0, 0], [chart.x1, chart.y0, 0]]} accent={C.muted} width={1} opacity={0.9} />
        <Label mono size={0.3} color={C.text2} position={[chart.x0, chart.y1 + 0.5, 0]}>
          {`LR: разгон ${chart.warmup} шагов до ${chart.peak} → cosine до ${chart.floor}`}
        </Label>
        {chart.ticks.map((tick) => (
          <group key={tick.epoch} position={[tick.x, chart.y0, 0]}>
            <Wire points={[[0, 0, 0], [0, -0.25, 0]]} accent={C.muted} width={1} opacity={0.9} />
            <Label mono size={0.28} anchorX="center" anchorY="top" color={C.text2} position={[0, -0.35, 0]}>
              {`эп. ${tick.epoch}`}
            </Label>
            {tick.val !== null ? (
              <Label mono size={0.28} anchorX="center" anchorY="top" color={C.text} position={[0, -0.85, 0]}>
                {num(tick.val, 3)}
              </Label>
            ) : null}
          </group>
        ))}
        <Label size={0.3} color={C.text2} position={[chart.x0, chart.y0 - 1.8, 0]}>
          {`${epochs} эпох · ${int(total)} шагов · val loss после каждой эпохи${run ? ` (прогон ${run.run})` : ''}`}
        </Label>
      </Appear>
    </group>
  )
}

export function LoopRegion() {
  return (
    <Region id="loop" size={[32, 13]}>
      <Content />
    </Region>
  )
}
