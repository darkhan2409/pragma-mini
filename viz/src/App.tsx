import { AnimatePresence, motion } from 'motion/react'
import { Suspense, lazy, useEffect } from 'react'
import { STEPS } from './content/steps'
import { useStore } from './store'
import { Stage } from './three/Stage'
import { Overlay } from './ui/Overlay'

// ============================================================
// ПРИЛОЖЕНИЕ
// ============================================================
//
// Сцена одна на всё время показа; 2D-экраны (шаги 14, 17, 18)
// грузятся отдельно и ложатся поверх, пока сцена не рисуется.
// Клавиши — как у презентации, кликер с PageUp/PageDown тоже.
// ============================================================

const SCREENS = {
  dashboard: lazy(() => import('./ui/screens/TrainingDashboard')),
  wave4: lazy(() => import('./ui/screens/Wave4Screen')),
  insight: lazy(() => import('./ui/screens/InsightScreen')),
}

function useKeys() {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)) return

      const s = useStore.getState()
      const present = s.mode === 'present'

      switch (event.key) {
        case 'ArrowRight':
        case 'PageDown':
        case ' ':
        case 'Enter':
          if (present) s.next()
          break
        case 'ArrowLeft':
        case 'PageUp':
        case 'Backspace':
          if (present) s.prev()
          break
        case 'Home':
          s.go(0)
          break
        case 'End':
          s.go(STEPS.length - 1)
          break
        case 's':
        case 'S':
        case 'ы':
        case 'Ы':
          s.togglePanel('steps')
          break
        case '?':
          s.togglePanel('help')
          break
        case 'e':
        case 'E':
        case 'у':
        case 'У':
          s.setMode('explore')
          break
        case 'p':
        case 'P':
        case 'з':
        case 'З':
          s.setMode('present')
          break
        case 'h':
        case 'H':
        case 'р':
        case 'Р':
          s.toggleCaptions()
          break
        case 'l':
        case 'L':
        case 'д':
        case 'Д':
          s.toggleLowPower()
          break
        case 'f':
        case 'F':
        case 'а':
        case 'А':
          if (document.fullscreenElement) void document.exitFullscreen()
          else void document.documentElement.requestFullscreen()
          break
        case 'r':
        case 'R':
        case 'к':
        case 'К':
          if (!present) s.overview()
          break
        case 'Escape':
          if (s.panel) s.togglePanel(s.panel)
          else if (s.selected) s.select(null)
          else if (!present) s.setMode('present')
          break
        default:
          return
      }
      event.preventDefault()
    }

    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}

export default function App() {
  useKeys()

  const screen = useStore((s) => (s.mode === 'present' ? STEPS[s.step].screen : undefined))
  const Screen = screen ? SCREENS[screen] : null

  return (
    <>
      <Stage />
      <AnimatePresence>
        {Screen ? (
          <motion.div
            key={screen}
            style={{ position: 'absolute', inset: 0 }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
          >
            <Suspense fallback={null}>
              <Screen />
            </Suspense>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <Overlay />
    </>
  )
}
