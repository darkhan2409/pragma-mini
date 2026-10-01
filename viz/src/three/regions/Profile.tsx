import { demo, num, shortValue } from '../../data/demo'
import { color as C } from '../../theme'
import { Appear, Arrow, Block, Glyph, Label, useRegionView } from '../primitives'
import { Region } from '../Region'

// ============================================================
// ШАГ 7. PROFILE ENCODER: АНКЕТА → ВЕКТОР НА МЕСТЕ [USR]
// ============================================================

function Content() {
  const { beat } = useRegionView()
  const tokens = demo.client.profile
  const top = 6.0
  const step = 0.64
  const encoder = demo.architecture.encoders.profile

  return (
    <group>
      {tokens.map((token, index) => {
        const y = top - index * step
        const marker = token.kind === 'special'
        const lifelong = token.key === 'profile_lifelong'
        const key = token.key.replace(/^profile_/, '')
        const text = marker ? '[USR]' : `${key} = ${shortValue(token).replace(/^profile_[a-z_]+_bucket_/, 'диапазон ')}`
        return (
          <Appear key={index} show delay={0.04 * index} position={[-14.2, y, 0]}>
            <Label mono size={0.34} color={marker ? C.amber : lifelong ? C.violet : C.text}>
              {text}
            </Label>
            {lifelong ? (
              <Label mono size={0.3} color={C.text2} position={[9.8, 0, 0]} anchorX="right">
                {`давность ${num(token.time_log ?? 0, 1)}`}
              </Label>
            ) : null}
          </Appear>
        )
      })}

      <Label size={0.32} color={C.text2} position={[-14.2, top - tokens.length * step - 0.2, 0]}>
        {`[USR] · ${demo.dataset.profile_fields.length} полей анкеты · вехи по времени`}
      </Label>

      <Appear show={beat >= 1}>
        <Arrow from={[-4.8, 1, 0]} to={[-2.9, 1, 0]} accent={C.amber} />
        <Block size={[2.2, 7.4, 1.2]} position={[-1.2, 1, 0]} accent={C.amber} />
        <Label size={0.36} anchorX="center" position={[-1.2, 5.3, 0]}>
          {`${encoder.blocks} блок · ${encoder.config.heads} головы`}
        </Label>
        <Label size={0.3} anchorX="center" color={C.violet} position={[-1.2, -3.2, 0]}>
          {'TimeRoPE: позиция вехи = давность до T'}
        </Label>
        <Arrow from={[0.4, 1, 0]} to={[3.4, 1, 0]} accent={C.amber} />
        <Glyph seed={4} accent={C.amber} position={[4.3, 1, 0]} height={4.2} width={0.46} label="вектор анкеты" caption="[128]" />
        <Label size={0.34} color={C.text2} position={[5.2, -0.2, 0]}>
          {'на месте [USR]'}
        </Label>
      </Appear>
    </group>
  )
}

export function ProfileRegion() {
  return (
    <Region id="profile" size={[32, 14]}>
      <Content />
    </Region>
  )
}
