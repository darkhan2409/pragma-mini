import type { IncomingMessage, ServerResponse } from 'node:http'
import { join } from 'node:path'
import type { Plugin } from 'vite'
import { readComplete } from './telemetryFile.ts'
import { TELEMETRY, dataDir, discoverRuns, wave4Statuses } from './runs.ts'

// ============================================================
// /api ДЛЯ ВИЗУАЛИЗАЦИИ (dev и preview)
// ============================================================
//
//   GET /api/runs                              прогоны с телеметрией
//   GET /api/telemetry?run=&offset=&identity=  целые новые строки
//   GET /api/wave4?ids=B0,N,E1                 статусы экспериментов
//
// Только чтение. run — id из найденного списка, путь из запроса
// не склеивается, поэтому выйти за data/ нельзя.
// ============================================================

type Next = (error?: unknown) => void

function send(response: ServerResponse, status: number, body: unknown) {
  response.statusCode = status
  response.setHeader('Content-Type', 'application/json; charset=utf-8')
  response.setHeader('Cache-Control', 'no-store')
  response.end(JSON.stringify(body))
}

export function handle(request: IncomingMessage, response: ServerResponse, next: Next): void {
  const url = new URL(request.url ?? '/', 'http://localhost')

  if (!url.pathname.startsWith('/api/')) {
    next()
    return
  }

  try {
    if (url.pathname === '/api/runs') {
      send(response, 200, { dataDir: dataDir(), runs: discoverRuns() })
      return
    }

    if (url.pathname === '/api/telemetry') {
      const id = url.searchParams.get('run') ?? ''
      const run = discoverRuns().find((item) => item.id === id)
      if (!run) {
        send(response, 404, { error: `нет прогона ${id}` })
        return
      }
      const offset = Number(url.searchParams.get('offset') ?? 0)
      const identity = url.searchParams.get('identity')
      send(response, 200, readComplete(join(run.path, TELEMETRY), Number.isFinite(offset) ? offset : 0, identity || null))
      return
    }

    if (url.pathname === '/api/wave4') {
      const ids = (url.searchParams.get('ids') ?? '')
        .split(',')
        .map((item) => item.trim())
        .filter((item) => /^[A-Za-z0-9]+$/.test(item))
      send(response, 200, { statuses: wave4Statuses(ids) })
      return
    }

    send(response, 404, { error: 'нет такого пути' })
  } catch (error) {
    send(response, 500, { error: String(error) })
  }
}

export function pragmaApi(): Plugin {
  return {
    name: 'pragma-api',
    configureServer(server) {
      server.middlewares.use(handle)
    },
    configurePreviewServer(server) {
      server.middlewares.use(handle)
    },
  }
}
