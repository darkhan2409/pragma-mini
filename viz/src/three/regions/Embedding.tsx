import { useMemo } from 'react'
import { demo, hero, int, shortValue, type TokenKind } from '../../data/demo'
import type { Vec3 } from '../../content/layout'
import { color as C } from '../../theme'
import { Appear, Card, Chip, Glyph, Label, Wire, chipWidth, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 4. ОБЩАЯ ТАБЛИЦА ЭМБЕДДИНГОВ И ВЕКТОРЫ ТОКЕНОВ
// ============================================================

const KIND_TITLE: Record<TokenKind, string> = {
  special: 'служебные',
  key: 'ключи',
  value: 'категории',
  bucket: 'диапазоны',
  bpe: 'куски BPE',
}

const ORDER: TokenKind[] = ['special', 'key', 'value', 'bucket', 'bpe']

// Высота полосы — от логарифма счёта: иначе 5 служебных не
// было бы видно рядом с 4096 кусками.
function bands(top: number, height: number) {
  const kinds = demo.vocabulary.kinds
  const weights = ORDER.map((kind) => Math.log10(kinds[kind].count + 3))
  const total = weights.reduce((a, b) => a + b, 0)
  let y = top
  return ORDER.map((kind, index) => {
    const h = (weights[index] / total) * height
    const band = { kind, top: y, bottom: y - h, mid: y - h / 2, ...kinds[kind] }
    y -= h
    return band
  })
}

const TABLE_X = -1.2
const TABLE_W = 4.2

function Content() {
  const { beat } = useRegionView()
  const layout = useMemo(() => bands(5.4, 10.4), [])
  const bandOf = (kind: TokenKind) => layout.find((band) => band.kind === kind)!

  // Токены примера: ключ названия и два куска BPE, ключ суммы и её диапазон.
  const picks = useMemo(() => {
    const name = hero.tokens.filter((token) => token.key === 'merchant_name')
    const amount = hero.tokens.find((token) => token.key === 'transaction_amount')!
    return [
      { text: `key:${name[0].key}`, kind: 'key' as TokenKind, id: name[0].key_id },
      ...name.map((token) => ({ text: `bpe:${shortValue(token)}`, kind: 'bpe' as TokenKind, id: token.value_id })),
      { text: amount.value.replace('bucket:transaction_amount_', 'bucket:…'), kind: 'bucket' as TokenKind, id: amount.value_id },
    ]
  }, [])

  const rowY = (kind: TokenKind, id: number) => {
    const band = bandOf(kind)
    const span = Math.max(1, band.last - band.first)
    return band.top - 0.15 - ((id - band.first) / span) * (band.top - band.bottom - 0.3)
  }

  return (
    <group>
      {/* Таблица */}
      <Card width={TABLE_W} height={10.8} position={[TABLE_X, 0.2, -0.05]} fill={C.surface} />
      {layout.map((band, index) => (
        <group key={band.kind}>
          {index > 0 ? (
            <Wire points={[[TABLE_X - TABLE_W / 2, band.top, 0], [TABLE_X + TABLE_W / 2, band.top, 0]]} accent={C.line} width={1} opacity={1} />
          ) : null}
          <Label size={0.36} color={C.text} position={[TABLE_X + TABLE_W / 2 + 0.4, band.mid + 0.2, 0]}>
            {`${KIND_TITLE[band.kind]} · ${int(band.count)}`}
          </Label>
          <Label mono size={0.28} color={C.muted} position={[TABLE_X + TABLE_W / 2 + 0.4, band.mid - 0.3, 0]}>
            {`${band.first}–${band.last}`}
          </Label>
        </group>
      ))}
      <Label size={0.46} anchorX="center" color={C.text} position={[TABLE_X, 6.2, 0]}>
        {`E: ${int(demo.vocabulary.size)} × ${demo.architecture.dim}`}
      </Label>

      {/* Токены и выбор строк */}
      {picks.map((pick, index) => {
        const y = 3.6 - index * 1.7
        const width = chipWidth(pick.text, 0.34)
        const from: Vec3 = [-9.6 + width, y, 0]
        const to: Vec3 = [TABLE_X - TABLE_W / 2 + 0.2, rowY(pick.kind, pick.id), 0]
        return (
          <Appear key={pick.text} show={beat >= 1} delay={0.15 * index}>
            <Chip text={pick.text} strong size={0.34} width={width} position={[-9.6 + width / 2, y, 0]} />
            <Label mono size={0.26} color={C.muted} position={[-9.6, y - 0.62, 0]}>
              {`строка ${pick.id}`}
            </Label>
            <Wire points={[from, [from[0] + 0.8, y, 0], [to[0] - 0.8, to[1], 0], to]} accent={C.cyan} width={1.2} opacity={0.75} />
            <mesh position={[TABLE_X, to[1], 0.01]}>
              <planeGeometry args={[TABLE_W - 0.3, 0.07]} />
              <meshBasicMaterial color={C.cyan} />
            </mesh>
          </Appear>
        )
      })}

      {/* Векторы токенов */}
      {picks.map((pick, index) => (
        <Appear key={`glyph-${pick.text}`} show={beat >= 2} delay={0.12 * index} position={[8.2 + index * 1.55, 0.6, 0]}>
          <Glyph seed={pick.id} height={4.4} caption={index === 0 ? '[128]' : null} />
        </Appear>
      ))}
      <Appear show={beat >= 2} delay={0.5} position={[8, 3.8, 0]}>
        <Label size={0.36} color={C.text2}>
          {'векторы токенов'}
        </Label>
      </Appear>
    </group>
  )
}

export function EmbeddingRegion() {
  return (
    <Region id="embedding" size={[34, 15]}>
      <Content />
    </Region>
  )
}
