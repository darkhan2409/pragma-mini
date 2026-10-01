import { useEffect, useRef, useState } from 'react'
import type { Telemetry } from './model'

// ============================================================
// ОПРОС ТЕЛЕМЕТРИИ
// ============================================================
//
// Раз в refresh секунд читатель дочитывает новое; вкладка в фоне —
// опрос на паузе. Пока идёт новый опрос, показывается прежний
// снимок: без мигания и скачков разметки.
// ============================================================

export interface Reader {
  poll: () => Promise<Telemetry>
}

export function useTelemetry(reader: Reader | null, refresh: number) {
  const [snapshot, setSnapshot] = useState<{ telemetry: Telemetry | null; version: number; error: string | null }>({
    telemetry: null,
    version: 0,
    error: null,
  })
  const busy = useRef(false)

  useEffect(() => {
    if (!reader) return

    let alive = true
    let timer: ReturnType<typeof setTimeout> | undefined

    const tick = async () => {
      if (!alive) return
      if (!document.hidden && !busy.current) {
        busy.current = true
        try {
          const telemetry = await reader.poll()
          if (alive) setSnapshot((prev) => ({ telemetry, version: prev.version + 1, error: null }))
        } catch (error) {
          if (alive) setSnapshot((prev) => ({ ...prev, error: String(error) }))
        } finally {
          busy.current = false
        }
      }
      timer = setTimeout(tick, refresh * 1000)
    }

    void tick()

    return () => {
      alive = false
      if (timer) clearTimeout(timer)
    }
  }, [reader, refresh])

  return snapshot
}
