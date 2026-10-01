import { create } from 'zustand'
import { STEPS } from './content/steps'

// ============================================================
// СОСТОЯНИЕ ПРЕЗЕНТАЦИИ
// ============================================================
//
// Режим, шаг и бит (подшаг по стрелке), выбранная деталь в
// EXPLORE. Положение повторяется в адресе (#/present/7/1): после
// перезагрузки страница вернётся на то же место.
// ============================================================

export type Mode = 'present' | 'explore'

interface State {
  mode: Mode
  step: number
  beat: number
  selected: string | null
  captions: boolean
  lowPower: boolean
  panel: 'steps' | 'help' | null
  // Счётчик запросов «общий вид» в EXPLORE (клавиша R).
  recenter: number
  next: () => void
  prev: () => void
  go: (step: number, beat?: number) => void
  setMode: (mode: Mode) => void
  select: (part: string | null) => void
  toggleCaptions: () => void
  toggleLowPower: () => void
  togglePanel: (panel: 'steps' | 'help') => void
  overview: () => void
}

const LAST = STEPS.length - 1

function clampBeat(step: number, beat: number): number {
  return Math.max(0, Math.min(beat, STEPS[step].beats.length - 1))
}

export function parseHash(hash: string): { mode: Mode; step: number; beat: number } {
  const [, mode, step, beat] = hash.replace(/^#/, '').split('/')
  const index = Math.max(0, Math.min(LAST, (Number(step) || 1) - 1))
  return {
    mode: mode === 'explore' ? 'explore' : 'present',
    step: index,
    beat: clampBeat(index, Number(beat) || 0),
  }
}

const initial = typeof window === 'undefined' ? parseHash('') : parseHash(window.location.hash)

const lowPowerFromUrl =
  typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('lowpower') === '1'

export const useStore = create<State>((set, get) => ({
  mode: initial.mode,
  step: initial.step,
  beat: initial.beat,
  selected: null,
  captions: true,
  lowPower: lowPowerFromUrl,
  panel: null,
  recenter: 0,

  next: () => {
    const { step, beat } = get()
    if (beat < STEPS[step].beats.length - 1) set({ beat: beat + 1 })
    else if (step < LAST) set({ step: step + 1, beat: 0 })
  },

  prev: () => {
    const { step, beat } = get()
    if (beat > 0) set({ beat: beat - 1 })
    else if (step > 0) set({ step: step - 1, beat: STEPS[step - 1].beats.length - 1 })
  },

  go: (step, beat = 0) => {
    const index = Math.max(0, Math.min(LAST, step))
    set({ step: index, beat: clampBeat(index, beat) })
  },

  setMode: (mode) => set({ mode, selected: null }),
  select: (selected) => set({ selected }),
  toggleCaptions: () => set({ captions: !get().captions }),
  toggleLowPower: () => set({ lowPower: !get().lowPower }),
  togglePanel: (panel) => set({ panel: get().panel === panel ? null : panel }),
  overview: () => set({ selected: null, recenter: get().recenter + 1 }),
}))

if (typeof window !== 'undefined') {
  useStore.subscribe((state) => {
    const hash = `#/${state.mode}/${state.step + 1}/${state.beat}`
    if (window.location.hash !== hash) window.history.replaceState(null, '', hash)
  })
}

// Бит шага для региона: в фокусе — текущий, иначе шаг показан
// целиком (последний бит).
export function beatFor(stepId: string, state: { step: number; beat: number }): number {
  const index = STEPS.findIndex((step) => step.id === stepId)
  if (index < 0) return 0
  if (index === state.step) return state.beat
  return index < state.step ? STEPS[index].beats.length - 1 : -1
}
