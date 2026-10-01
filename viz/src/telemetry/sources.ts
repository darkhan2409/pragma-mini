import { addLines, emptyTelemetry, type Telemetry } from './model'

// ============================================================
// ИСТОЧНИКИ ТЕЛЕМЕТРИИ
// ============================================================
//
//   live  /api/telemetry dev-сервера: целые новые строки с места
//         остановки, reset при новом файле;
//   file  выбранный файл: в Chromium — File System Access с
//         опросом, иначе один раз;
//   mock  синтетический прогон только для разработки: экран
//         помечает его «СИНТЕТИКА».
// ============================================================

export interface RunEntry {
  id: string
  telemetry: boolean
  size: number
  mtimeMs: number
}

export async function listRuns(): Promise<RunEntry[]> {
  const response = await fetch('/api/runs')
  if (!response.ok) throw new Error(`/api/runs: ${response.status}`)
  const body = (await response.json()) as { runs: RunEntry[] }
  return body.runs
}

interface Chunk {
  missing?: true
  identity: string
  size: number
  mtimeMs: number
  reset: boolean
  offset: number
  next: number
  text: string
}

// Читатель одного прогона через API: дочитывает с места остановки.
export class LiveReader {
  telemetry: Telemetry = emptyTelemetry()
  private offset = 0
  private identity: string | null = null

  constructor(readonly run: string) {}

  async poll(): Promise<Telemetry> {
    for (let round = 0; round < 16; round++) {
      const query = new URLSearchParams({ run: this.run, offset: String(this.offset), identity: this.identity ?? '' })
      const response = await fetch(`/api/telemetry?${query}`)
      if (!response.ok) throw new Error(`/api/telemetry: ${response.status}`)
      const chunk = (await response.json()) as Chunk

      if (chunk.missing) {
        if (this.offset > 0 || this.telemetry.runs.length) this.reset()
        return this.telemetry
      }

      if (chunk.reset) this.reset()

      this.identity = chunk.identity
      this.telemetry.modified = chunk.mtimeMs / 1000

      if (chunk.text) addLines(this.telemetry, chunk.text)
      this.offset = chunk.next

      if (chunk.next >= chunk.size || !chunk.text) break
    }
    return this.telemetry
  }

  private reset() {
    this.telemetry = emptyTelemetry()
    this.offset = 0
  }
}

// Файл с диска: целые строки с места остановки.
export class FileReader_ {
  telemetry: Telemetry = emptyTelemetry()
  private offset = 0
  private stamp = 0

  constructor(
    private readonly source: File | FileSystemFileHandle,
  ) {}

  get name(): string {
    return this.source.name
  }

  async poll(): Promise<Telemetry> {
    const file = 'getFile' in this.source ? await this.source.getFile() : this.source

    if (file.size < this.offset || (this.stamp && file.lastModified < this.stamp)) {
      this.telemetry = emptyTelemetry()
      this.offset = 0
    }

    this.stamp = file.lastModified
    this.telemetry.modified = file.lastModified / 1000

    if (file.size > this.offset) {
      const text = await file.slice(this.offset).text()
      const end = text.lastIndexOf('\n')
      if (end >= 0) {
        addLines(this.telemetry, text.slice(0, end + 1))
        this.offset += new TextEncoder().encode(text.slice(0, end + 1)).length
      }
    }

    return this.telemetry
  }
}

// Синтетический прогон: те же виды записей, растёт со временем.
export class MockReader {
  telemetry: Telemetry = emptyTelemetry()
  private step = 0
  private readonly perEpoch = 300
  private readonly epochs = 3

  async poll(): Promise<Telemetry> {
    const t = this.telemetry
    if (!t.runs.length) {
      addLines(
        t,
        JSON.stringify({ kind: 'run', epoch: 1, step: 0, epochs: this.epochs, planned_epochs: this.epochs, total_steps: this.perEpoch * this.epochs, max_steps: null, max_grad_norm: 1, resumed: false }),
      )
    }
    const total = this.perEpoch * this.epochs
    const lines: string[] = []
    for (let i = 0; i < 25 && this.step < total; i++) {
      this.step += 1
      const epoch = Math.ceil(this.step / this.perEpoch)
      const loss = 1.9 + 6 * Math.exp(-this.step / 60) + 0.15 * Math.sin(this.step * 1.7)
      const lr = this.step < 30 ? (3e-4 * this.step) / 30 : 3e-5 + (3e-4 - 3e-5) * 0.5 * (1 + Math.cos((Math.PI * (this.step - 30)) / (total - 30)))
      lines.push(JSON.stringify({ kind: 'step', epoch, step: this.step, loss, lr, grad_norm: 0.5 + 0.4 * Math.abs(Math.sin(this.step)), tokens: 38000, seconds: 0.3, targets: 9000 }))
      if (this.step % this.perEpoch === 0) {
        lines.push(JSON.stringify({ kind: 'epoch', epoch, step: this.step, train_loss: loss + 0.1, val_loss: 2.1 - 0.08 * epoch, learning_rate: lr, cuda_peak_allocated_gib: 1.9, cuda_peak_reserved_gib: 2.3 }))
      }
    }
    addLines(t, lines.join('\n'))
    t.modified = Date.now() / 1000
    return t
  }
}
