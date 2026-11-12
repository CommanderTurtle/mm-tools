import { Eye, MoreHorizontal, Sparkles, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import type { BackendConfig } from '../types'

interface ResourceStatusProps {
  config?: BackendConfig
  dark?: boolean
}

const gib = (mib?: number) => `${((mib ?? 0) / 1024).toFixed(1)} GiB`
const percent = (ratio?: number) => `${Math.round((ratio ?? 0) * 100)}%`

export function ResourceStatus({ config, dark = false }: ResourceStatusProps) {
  const [open, setOpen] = useState(false)
  const resources = config?.resources
  const errors = config?.recent_errors ?? []
  const memoryPressure = resources?.memory_pressure ?? 'normal'
  const nodeTotal = config?.gpu_node_total
  const nodeAvailable = config?.gpu_node_available
  const activeUsers = config?.dispatcher_active_users ?? 0

  useEffect(() => {
    if (!open) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [open])

  return (
    <div className={`resource-status ${dark ? 'dark' : ''}`}>
      <button
        className="resource-status-trigger"
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-label="Open live Demo resource status"
      >
        <MoreHorizontal size={14} />
        <span>
          {nodeTotal !== undefined
            ? `${activeUsers} / ${nodeTotal} active users`
            : 'Status'}
        </span>
        {memoryPressure !== 'normal' && (
          <Sparkles className={`memory-${memoryPressure}`} size={13} />
        )}
      </button>

      {open && createPortal(
        <>
          <button
            className="resource-panel-backdrop"
            type="button"
            onClick={() => setOpen(false)}
            aria-label="Close resource status"
          />
          <aside className="resource-panel" aria-label="Demo resource status">
          <header>
            <div>
              <small>LIVE CONTROL PLANE</small>
              <strong>Demo status</strong>
            </div>
            <button type="button" onClick={() => setOpen(false)} aria-label="Close resource status">
              <X size={16} />
            </button>
          </header>

          <div className="resource-summary-grid">
            <div>
              <span>Online</span>
              <strong>{config?.online_users ?? 0}</strong>
              <small>active browsers</small>
            </div>
            <div>
              <span>Queue</span>
              <strong>{config?.queued_jobs ?? 0}</strong>
              <small>{config?.running_jobs ?? 0} running</small>
            </div>
            <div>
              <span>{nodeTotal !== undefined ? 'SP16 slots' : 'Workers'}</span>
              <strong>
                {nodeTotal !== undefined
                  ? `${activeUsers} / ${nodeTotal}`
                  : config?.online_worker_count ?? 0}
              </strong>
              <small>
                {nodeTotal !== undefined
                  ? `${nodeAvailable ?? 0} available · ${config?.dispatcher_waitlist_size ?? 0} waiting`
                  : `${config?.reserved_session_count ?? 0} pinned sessions`}
              </small>
            </div>
            <div>
              <span>Compute</span>
              <strong>{resources?.compute_state ?? 'unknown'}</strong>
              <small>
                {Math.round(resources?.average_gpu_utilization_percent ?? 0)}% SM · {percent(resources?.average_power_ratio)} power
              </small>
            </div>
          </div>

          <section className="resource-section">
            <div className="resource-section-title">
              <Eye size={14} />
              <span>GPU memory and power</span>
              <em className={`memory-${memoryPressure}`}>{memoryPressure}</em>
            </div>
            {!resources?.available ? (
              <p className="resource-empty">GPU telemetry is currently unavailable.</p>
            ) : (
              <div className="gpu-resource-list">
                {resources.gpus.map((gpu) => {
                  const memoryRatio = gpu.memory_total_mib
                    ? gpu.memory_used_mib / gpu.memory_total_mib
                    : 0
                  return (
                    <div className="gpu-resource-row" key={gpu.index}>
                      <div>
                        <strong>GPU {gpu.index}</strong>
                        <span>{gib(gpu.memory_used_mib)} / {gib(gpu.memory_total_mib)}</span>
                      </div>
                      <div className="gpu-memory-track">
                        <i style={{ width: percent(memoryRatio) }} />
                      </div>
                      <small>
                        SM {Math.round(gpu.gpu_utilization_percent)}% · {Math.round(gpu.power_draw_watts)}W/{Math.round(gpu.power_limit_watts)}W
                      </small>
                    </div>
                  )
                })}
              </div>
            )}
          </section>

          <section className="resource-section">
            <div className="resource-section-title">
              <Sparkles size={14} />
              <span>Recent recoverable errors</span>
            </div>
            {errors.length === 0 ? (
              <p className="resource-empty">No recent inference errors.</p>
            ) : (
              <div className="resource-error-list">
                {errors.map((error) => (
                  <article key={error.id}>
                    <strong>{error.code}</strong>
                    <span>{error.message}</span>
                    <small>
                      {new Date(error.created_at * 1000).toLocaleString()}
                      {error.job_id ? ` · ${error.job_id}` : ''}
                    </small>
                  </article>
                ))}
              </div>
            )}
          </section>

          <footer>
            “Active” requires a running Demo job plus material SM or power draw;
            a standalone 100% nvidia-smi sample is not treated as proof of useful compute.
          </footer>
          </aside>
        </>,
        document.body,
      )}
    </div>
  )
}
