import { useMemo } from 'react'
import { bucketNumber, candidates, type Candidate } from '../../data/checkpoint'
import { demo, hero, int, num, shortValue, type BucketRange } from '../../data/demo'
import { MASK_TEXT } from '../../content/steps'
import { color as C } from '../../theme'
import { Appear, Arrow, Block, Card, Chip, Glyph, Label, chipWidth, useRegionView } from '../primitives'
import { Region } from '../Region'
import { HERO_ROWS, tokenText } from './Tokens'

// ============================================================
// ШАГ 11. MLM: СКРЫТОЕ ЗНАЧЕНИЕ → ТРИ ВЕКТОРА → ЛОГИТЫ
// ============================================================
//
// Маскируется сумма сквозного события, цель — настоящий токен.
// Строки предсказания — из data/checkpoint.ts: с чекпойнтом top-5
// по всему словарю и цель с настоящими вероятностями, без него —
// схема из пяти диапазонов вокруг цели.
// ============================================================

function note(item: Candidate): string {
  const mlm = demo.model?.mlm
  if (!mlm) return `${Math.round(item.p * 100)}%${item.target ? '  цель' : ''}`
  const percent = `${num(item.p * 100, 1)}%`
  if (item.target) return item.top ? `${percent}  цель = ответ модели` : `${percent}  цель · ${mlm.target_rank}-е место`
  return item.top ? `${percent}  ответ модели` : percent
}

function span(bucket: BucketRange): string {
  const left = bucket.min === null ? '<' : int(bucket.min)
  return bucket.max === null ? `≥ ${left}` : bucket.min === null ? `< ${int(bucket.max)}` : `${left}–${int(bucket.max)}`
}

function Content() {
  const { beat } = useRegionView()
  const top = useMemo(candidates, [])
  const peak = Math.max(...top.map((item) => item.p))

  return (
    <group>
      {/* Событие с маской */}
      <Card width={11.6} height={6.6} position={[-9.4, 2.6, -0.05]} />
      <Label size={0.4} color={C.text2} position={[-14.8, 5.4, 0]}>
        {'событие под маской'}
      </Label>
      {HERO_ROWS.map((row, index) => {
        const masked = row.key === 'transaction_amount'
        const y = 4.4 - index * 1.25
        let x = -9.4
        return (
          <group key={row.key} position={[0, y, 0]}>
            <Label mono size={0.32} color={C.text2} anchorX="right" position={[-9.8, 0, 0]}>
              {row.key}
            </Label>
            {masked ? (
              <Chip text="[MASK]" accent={C.rose} strong position={[-9.4 + chipWidth('[MASK]', 0.34) / 2, 0, 0]} size={0.34} textColor={C.rose} />
            ) : (
              row.tokens.map((token, i) => {
                const text = token.kind === 'bpe' ? shortValue(token) : tokenText(token).split('=').pop()!
                const width = chipWidth(text, 0.34)
                const at = x + width / 2
                x += width + 0.2
                return <Chip key={i} text={text} position={[at, 0, 0]} size={0.34} width={width} />
              })
            )}
          </group>
        )
      })}
      <Label size={0.3} color={C.text2} position={[-14.8, -1.2, 0]}>
        {`ключ виден, значение скрыто · ${MASK_TEXT}`}
      </Label>

      {/* Три входа головы */}
      <Appear show={beat >= 1}>
        {[
          { label: 'токен', note: 'Event Encoder', accent: C.cyan, seed: 1623 },
          { label: 'событие', note: 'History', accent: C.cyan, seed: hero.index + 900 },
          { label: '[USR]', note: 'History', accent: C.amber, seed: 4 },
        ].map((input, index) => (
          <group key={input.label} position={[-3.3 + index * 2.1, 2.4, 0]}>
            <Glyph seed={input.seed} accent={input.accent} height={3.0} width={0.38} caption={null} label={input.label} />
            <Label size={0.24} anchorX="center" color={C.text2} position={[0, -1.95, 0]}>
              {input.note}
            </Label>
          </group>
        ))}
        <Arrow from={[1.4, 2.4, 0]} to={[2.5, 2.4, 0]} accent={C.muted} />
        {[C.cyan, C.cyan, C.amber].map((tint, index) => (
          <mesh key={index} position={[3.2 + index * 0.52, 2.4, 0]}>
            <planeGeometry args={[0.46, 3.0]} />
            <meshBasicMaterial color={tint} transparent opacity={0.55} />
          </mesh>
        ))}
        <Label mono size={0.3} anchorX="center" color={C.text2} position={[3.72, 0.55, 0]}>
          {`concat [${3 * demo.architecture.dim}]`}
        </Label>
        <Arrow from={[4.5, 2.4, 0]} to={[5.4, 2.4, 0]} accent={C.muted} />
        <Block size={[2.2, 2.0, 1]} position={[6.6, 2.4, 0]} accent={C.cyan} />
        <Label mono size={0.3} anchorX="center" position={[6.6, 2.4, 0.6]}>
          {`Linear ${3 * demo.architecture.dim}→${demo.architecture.dim}`}
        </Label>
        <Arrow from={[7.8, 2.4, 0]} to={[8.7, 2.4, 0]} accent={C.muted} />
        <Glyph seed={77} position={[9.2, 2.4, 0]} height={3.0} width={0.38} caption="h [128]" />
      </Appear>

      {/* Логиты той же таблицей и кандидаты */}
      <Appear show={beat >= 2}>
        <Arrow from={[9.8, 2.4, 0]} to={[11.2, 2.4, 0]} accent={C.cyan} />
        <Label size={0.32} color={C.cyan} position={[11.5, 2.65, 0]}>
          {'× E^T'}
        </Label>
        <Label size={0.28} color={C.text2} position={[11.5, 2.05, 0]}>
          {'та же таблица'}
        </Label>
        <Label size={0.3} color={C.text2} position={[4.0, -1.2, 0]}>
          {demo.model
            ? `${demo.client.mlm_candidates.key}, direction = ${String(hero.raw.direction)}: пять самых вероятных токенов словаря и цель`
            : `${demo.client.mlm_candidates.key}: диапазоны при direction = ${String(hero.raw.direction)} · вероятности — схема`}
        </Label>
        {top.map((item, index) => {
          const y = -2.0 - index * 0.56
          const width = (2.6 * item.p) / peak
          const tint = item.target ? C.green : item.top ? C.cyan : C.muted
          return (
            <group key={item.value_id} position={[4.0, y, 0]}>
              <Label mono size={0.3} color={item.target ? C.green : item.top ? C.text : C.text2}>
                {item.bucket ? `bucket_${bucketNumber(item.bucket.name)}  ${span(item.bucket)} KZT` : item.name}
              </Label>
              <mesh position={[5.2 + width / 2, 0, 0]}>
                <planeGeometry args={[width, 0.34]} />
                <meshBasicMaterial color={tint} />
              </mesh>
              <Label mono size={0.3} color={item.target ? C.green : item.top ? C.text : C.text2} position={[5.4 + width, 0, 0]}>
                {note(item)}
              </Label>
            </group>
          )
        })}
      </Appear>
    </group>
  )
}

export function MlmRegion() {
  return (
    <Region id="mlm" size={[32, 14]}>
      <Content />
    </Region>
  )
}
