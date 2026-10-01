import { demo, hero, num, pairs, range, shortValue, type Token } from '../../data/demo'
import { color as C } from '../../theme'
import { Appear, Arrow, Chip, Label, chipWidth, useRegionView } from '../primitives'
import { Region } from '../Region'
import { MAIN_KEYS } from './Event'

// ============================================================
// ШАГ 3. ТОКЕНИЗАЦИЯ: ПАРА → НОМЕРА СЛОВАРЯ И ПОЗИЦИИ
// ============================================================

const KIND_NOTE: Record<string, string> = {
  value: 'категория',
  bucket: 'диапазон суммы',
  bpe: 'куски BPE',
}

export const HERO_ROWS = pairs(hero).filter((row) => MAIN_KEYS.includes(row.key))

// Колонки: левые края.
const RAW_X = -15.4
const KEY_X = -8.0
const VALUE_X = -1.6
const IDS_X = 8.6
const SIZE = 0.32

export function tokenText(token: Token): string {
  return token.kind === 'bpe' ? `bpe:${shortValue(token)}` : token.value
}

function ValueChips({ tokens, show, delay }: { tokens: Token[]; show: boolean; delay: number }) {
  let x = VALUE_X
  return (
    <>
      {tokens.map((token, index) => {
        const text = tokenText(token)
        const width = chipWidth(text, SIZE)
        const at = x + width / 2
        x += width + 0.25
        return (
          <Appear key={`${token.value_id}-${index}`} show={show} delay={delay + index * 0.12} position={[at, 0, 0]}>
            <Chip text={text} strong width={width} size={SIZE} />
          </Appear>
        )
      })}
    </>
  )
}

function Content() {
  const { beat } = useRegionView()
  const top = 4.4
  const step = 2.0
  const marker = hero.tokens[0]

  const rows = [{ key: '[EVT]', tokens: [marker] }, ...HERO_ROWS]

  return (
    <group>
      <Label size={0.4} color={C.text2} position={[RAW_X, top + 1.6, 0]}>
        {'пара ключ = значение'}
      </Label>
      <Appear show={beat >= 1} position={[KEY_X, top + 1.6, 0]}>
        <Label size={0.4} color={C.text2}>
          {'ключ'}
        </Label>
      </Appear>
      <Appear show={beat >= 1} position={[VALUE_X, top + 1.6, 0]}>
        <Label size={0.4} color={C.text2}>
          {'значение'}
        </Label>
      </Appear>
      <Appear show={beat >= 2} position={[IDS_X, top + 1.6, 0]}>
        <Label size={0.4} color={C.text2}>
          {'номера key · value, позиция'}
        </Label>
      </Appear>

      {rows.map((row, index) => {
        const y = top - index * step
        const first = row.tokens[0]
        const marker = row.key === '[EVT]'
        const raw = marker ? '[EVT]' : `${row.key} = ${String(hero.raw[row.key])}`
        const note = marker ? 'маркер: один номер в обоих слотах' : (KIND_NOTE[first.kind] ?? '')
        const bucket = first.kind === 'bucket' && first.range ? ` ${range(first.range)}` : ''
        const keyText = `key:${row.key}`
        const keyWidth = chipWidth(keyText, SIZE)

        return (
          <group key={row.key} position={[0, y, 0]}>
            <Label mono size={0.32} color={marker ? C.cyan : C.text} position={[RAW_X, 0, 0]}>
              {raw}
            </Label>

            <Appear show={beat >= 1} delay={0.1 * index}>
              <Arrow from={[KEY_X - 0.95, 0, 0]} to={[KEY_X - 0.2, 0, 0]} />
              {marker ? (
                <Chip text="[EVT]" accent={C.cyan} strong position={[KEY_X + chipWidth('[EVT]', SIZE) / 2, 0, 0]} size={SIZE} />
              ) : (
                <Chip text={keyText} accent={C.muted} position={[KEY_X + keyWidth / 2, 0, 0]} size={SIZE} width={keyWidth} />
              )}
              <Label size={0.28} color={C.muted} position={[marker ? KEY_X : VALUE_X, -0.66, 0]}>
                {note + bucket}
              </Label>
            </Appear>

            {marker ? null : <ValueChips tokens={row.tokens} show={beat >= 1} delay={0.25 + 0.1 * index} />}

            <Appear show={beat >= 2} delay={0.08 * index} position={[IDS_X, 0, 0]}>
              <Label mono size={0.36} color={C.text}>
                {row.tokens.map((token) => `${token.key_id}·${token.value_id}`).join('   ')}
              </Label>
              <Label mono size={0.28} color={C.violet} position={[0, -0.56, 0]}>
                {`позиция ${row.tokens.map((token) => token.position).join(', ')}`}
              </Label>
            </Appear>
          </group>
        )
      })}

      <Appear show={beat >= 2} delay={0.6} position={[RAW_X, top - rows.length * step + 0.7, 0]}>
        <Label size={0.36} color={C.text2}>
          {`${hero.tokens.length} токенов в этом событии · в среднем ${num(demo.client.n_tokens / demo.client.n_events, 1)} на событие клиента · словарь ${demo.vocabulary.size}`}
        </Label>
      </Appear>
    </group>
  )
}

export function TokensRegion() {
  return (
    <Region id="tokens" size={[34, 15]}>
      <Content />
    </Region>
  )
}
